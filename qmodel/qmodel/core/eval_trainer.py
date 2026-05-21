import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Iterable, Protocol, TypeAlias
import time
from tqdm import tqdm

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.profiler import MyProfiler
from qmodel.components.lr_scheduler import Scheduler
from qmodel.components.timer import RollingTimer

# alias torch cuda event and _cudaeventbase by the same one
Event: TypeAlias = torch.cuda.Event | torch._C._CudaEventBase
Stream: TypeAlias = torch.cuda.Stream | torch._C._CudaStreamBase


class _AsyncBase:
    device: torch.device
    move_target: bool
    model: nn.Module
    dataloader: Iterable | torch.utils.data.DataLoader
    profiler: MyProfiler
    amp_scaler: MyScaler
    valid: bool

    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, move_target: bool,
        timer_window_size: int,
        grad_clip_norm: float | None,
    ):
        # Initialize core async runtime state and streams.
        self.device = torch.device(device)
        self.move_target = move_target

        self.model = model
        self.dataloader = dataloader
        self.it = iter(dataloader)
        self.amp_scaler = amp_scaler

        # Initialize executors and streams used for overlap.
        self.ex = ThreadPoolExecutor(max_workers=1)
        self.ex2 = ThreadPoolExecutor(max_workers=1)
        self.stream1: Stream = torch.cuda.Stream(device=self.device)   # host->device
        self.stream2: Stream = torch.cuda.Stream(device=self.device)   # device->host
        self.model_stream: Stream = torch.cuda.Stream(device=self.device)

        # Initialize profiling and timing helpers.
        self.profiler = profiler
        self.timer = RollingTimer(timer_window_size)
        self.valid = True
        self.grad_clip_norm = grad_clip_norm

    def _h2d_transfer(self, curr_it: int):
        """Fetch one CPU batch and submit its host-to-device transfer."""
        # Measure Python/DataLoader time separately from CUDA copy time.
        loader_t0 = time.perf_counter()
        with torch.profiler.record_function(f"_h2d_transfer_{curr_it}"):
            try:
                data, target, other_meta = next(self.it)
            except StopIteration:
                return torch.Tensor(), torch.Tensor(), None, None, 0.0, 0.0, None, None
            loader_cpu_ms = (time.perf_counter() - loader_t0) * 1000.0

            # Submit non-blocking H2D copies on the transfer stream.
            h2d_submit_t0 = time.perf_counter()
            with torch.cuda.stream(self.stream1):  # type: ignore
                assert data.is_pinned()
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record(self.stream1)
                data = data.to(self.device, non_blocking=True)
                if self.move_target:
                    assert target.is_pinned()
                    target = target.to(self.device, non_blocking=True)
                h2d_end.record(self.stream1)
            h2d_submit_ms = (time.perf_counter() - h2d_submit_t0) * 1000.0

            # Return the completion event that the model stream must wait on.
            event = torch.cuda.Event()
            event.record(self.stream1)

            return data, target, other_meta, event, loader_cpu_ms, h2d_submit_ms, h2d_start, h2d_end

    def _init(self):
        raise NotImplementedError

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        raise NotImplementedError

    def _step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        return self._normal_step(data, target, other_meta, curr_it)

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

    def run(self, n_iter=99999999, start=0, use_tqdm=True) -> list:
        assert self.valid, "Can only use once, re-init class"
        self.valid = False
        self._init()

        # Start profiling and ensure the device is in a clean synchronized state.
        self.profiler.start(sleep_sec=2)
        torch.cuda.synchronize()

        res, gbuffer = [], None
        event2, event3 = torch.cuda.Event(), torch.cuda.Event()
        one_time_events = [torch.cuda.Event() for _ in range(3)]
        f1 = self.ex.submit(self._h2d_transfer, 0)

        if use_tqdm:
            iters = tqdm(range(start, n_iter), initial=start, miniters=20)
        else:
            iters = range(start, n_iter)
        for curr_it in iters:
            # Measure per-iteration CPU wall time for iter_ms.
            iter_t0 = time.perf_counter()
            self._pre_step(curr_it)
            checkpoint_ms = float(getattr(self, "_last_pre_step_ms", 0.0))
            self._last_pre_step_ms = 0.0
            data, target, other_meta, event1, loader_cpu_ms, h2d_submit_ms, h2d_start, h2d_end = f1.result()
            if event1 is None:  # means finished
                break

            # Measure GPU stall while waiting for H2D to make data ready.
            data_wait_start = torch.cuda.Event(enable_timing=True)
            data_wait_end = torch.cuda.Event(enable_timing=True)
            data_wait_start.record(self.model_stream)
            event1.wait(self.model_stream)
            data_wait_end.record(self.model_stream)

            # prev round gbuffer must have completed since they are in same stream
            # Measure GPU time spent running the model step on model_stream.
            model_run_start = torch.cuda.Event(enable_timing=True)
            model_run_end = torch.cuda.Event(enable_timing=True)
            model_run_start.record(self.model_stream)
            with torch.cuda.stream(self.model_stream):  # type: ignore
                output = self._step(data, target, other_meta, curr_it)
            model_run_end.record(self.model_stream)
            train_detail_events = getattr(self, "_last_train_detail_events", None)

            bs = output.shape[0]
            if gbuffer is None:
                gbuffer = torch.empty_like(output)
            event3.wait(self.stream1)

            # loop idx+1: start data transfer
            f1 = self.ex.submit(self._h2d_transfer, curr_it + 1)

            # wait idx-1 to prevent to fast CPU submit of tasks
            event2.synchronize()

            # loop idx-1: wait for result transfer
            event2.wait(self.model_stream)
            with torch.cuda.stream(self.model_stream):  # type: ignore
                gbuffer[:bs].copy_(output, non_blocking=True)
            event3.record(self.model_stream)  # event3: complete copy to gbuffer, can reuse output

            event3.wait(self.stream2)
            buffer = torch.empty_like(output, pin_memory=True, device="cpu")
            with torch.cuda.stream(self.stream2):  # type: ignore
                buffer.copy_(gbuffer[:bs], non_blocking=True)
            event2.record(self.stream2)    # event2: complete copy to host, can reuse gbuffer

            # one-time event used in async post-step
            event4 = one_time_events[curr_it % len(one_time_events)]
            # it should have been synced, but for safety we sync again
            event4.synchronize()
            event4.record(self.stream2)

            # Submit post-step work after D2H completes, and update timer from completed CUDA events.
            iter_ms = (time.perf_counter() - iter_t0) * 1000.0
            self.ex2.submit(
                self._post_step_async,
                res,
                buffer,
                target,
                other_meta,
                curr_it,
                event4,
                iter_ms,
                data_wait_start,
                data_wait_end,
                model_run_start,
                model_run_end,
                loader_cpu_ms,
                h2d_submit_ms,
                h2d_start,
                h2d_end,
                train_detail_events,
                checkpoint_ms,
            )
            self.profiler.step()

        torch.cuda.synchronize()
        self.profiler.stop()
        self.ex.shutdown(wait=True)
        self.ex2.shutdown(wait=True)

        return res

    def _post_step_async(
        self,
        res,
        buffer,
        target,
        other_meta,
        curr_it,
        event: Event,
        iter_ms: float,
        data_wait_start: Event,
        data_wait_end: Event,
        model_run_start: Event,
        model_run_end: Event,
        loader_cpu_ms: float,
        h2d_submit_ms: float,
        h2d_start: Event | None,
        h2d_end: Event | None,
        train_detail_events,
        checkpoint_ms: float,
    ):
        """(in sub-thread) ex2 will execute _post_step after event is recorded (i.e., buffer copy done)

        we need this method and the thread because we need to ensure data has arrived at buffer in cpu
        before we do anything with it
        """

        # Wait for D2H completion so CUDA timing queries do not introduce extra synchronization.
        event.synchronize()

        # Convert CUDA wait intervals into ms and push into the rolling timer.
        data_ms = float(data_wait_start.elapsed_time(data_wait_end))
        model_ms = float(model_run_start.elapsed_time(model_run_end))
        h2d_gpu_ms = float(h2d_start.elapsed_time(h2d_end)) if h2d_start is not None and h2d_end is not None else 0.0
        forward_ms, backward_ms, optimizer_ms = _train_detail_elapsed_ms(train_detail_events)
        self.timer.add(
            iter_ms=iter_ms,
            data_ms=data_ms,
            model_ms=model_ms,
            loader_cpu_ms=float(loader_cpu_ms),
            h2d_submit_ms=float(h2d_submit_ms),
            h2d_gpu_ms=float(h2d_gpu_ms),
            forward_ms=float(forward_ms),
            backward_ms=float(backward_ms),
            optimizer_ms=float(optimizer_ms),
            checkpoint_ms=float(checkpoint_ms),
        )

        with torch.profiler.record_function(f"_post_step_{curr_it}"):
            self._post_step(res, buffer, target, other_meta, curr_it)


class _SyncBase:
    device: torch.device
    move_target: bool
    model: nn.Module
    dataloader: Iterable | torch.utils.data.DataLoader
    profiler: MyProfiler
    amp_scaler: MyScaler
    valid: bool

    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, move_target: bool,
        timer_window_size: int,
        grad_clip_norm: float | None,
    ):
        # Initialize core sync runtime state and model stream.
        self.device = torch.device(device)
        self.move_target = move_target
        self.model = model
        self.dataloader = dataloader
        self.amp_scaler = amp_scaler
        self.model_stream: Stream = torch.cuda.Stream(device=self.device)

        # Initialize profiling and timing helpers.
        self.profiler = profiler
        self.timer = RollingTimer(timer_window_size)
        self.valid = True
        self.grad_clip_norm = grad_clip_norm

    def _init(self):
        raise NotImplementedError

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        raise NotImplementedError

    def _step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        return self._normal_step(data, target, other_meta, curr_it)

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError

    def run(self, n_iter=99999999, start=0, use_tqdm=True) -> list:
        assert self.valid, "Can only use once, re-init class"
        self.valid = False
        self._init()

        # Start profiling and ensure the device is in a clean synchronized state.
        self.profiler.start(sleep_sec=2)
        torch.cuda.synchronize()

        res = []
        it = iter(self.dataloader)

        if use_tqdm:
            iters = tqdm(range(start, n_iter), initial=start, miniters=20)
        else:
            iters = range(start, n_iter)
        for curr_it in iters:
            # Measure per-iteration CPU wall time for iter_ms.
            iter_t0 = time.perf_counter()
            data, target, other_meta = next(it)
            # print(data.ravel()[:3], target.ravel()[:3], other_meta[:3])
            self._pre_step(curr_it)
            data = data.to(self.device)
            if self.move_target:
                target = target.to(self.device)
            data_time_end = time.perf_counter()

            # Run the model step on the model stream.
            with torch.cuda.stream(self.model_stream):  # type: ignore
                output = self._step(data, target, other_meta, curr_it)
            # Measure CPU-visible wait for the device to finish the model step.
            torch.cuda.synchronize()
            model_time_end = time.perf_counter()
            buffer = output.cpu()

            self._post_step(res, buffer, target, other_meta, curr_it)
            self.profiler.step()

            # Push this step's timings into the rolling timer.
            iter_ms = (time.perf_counter() - iter_t0) * 1000.0
            data_ms = (data_time_end - iter_t0) * 1000.0
            model_ms = (model_time_end - data_time_end) * 1000.0
            self.timer.add(iter_ms=iter_ms, data_ms=data_ms, model_ms=model_ms)

        torch.cuda.synchronize()
        self.profiler.stop()

        return res


class InferenceProto(Protocol):
    model: nn.Module
    device: torch.device
    amp_scaler: MyScaler

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        ...


class TrainProto(Protocol):
    optimizer: torch.optim.Optimizer
    device: torch.device
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    lr_sched: Scheduler
    amp_scaler: MyScaler
    model: nn.Module
    grad_clip_norm: float | None

    def _normal_step(self, data, target, other_meta, curr_it) -> torch.Tensor:
        ...


class InferenceMixin:
    def _init(self: InferenceProto):
        self.model.eval()

    def _normal_step(self: InferenceProto, data, target, other_meta, curr_it) -> torch.Tensor:
        with torch.no_grad(), self.amp_scaler.autocast():
            output = self.model(data)
        return output

    def _step(self: InferenceProto, data, target, other_meta, curr_it) -> torch.Tensor:
        return self._normal_step(data, target, other_meta, curr_it)

    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self: InferenceProto, res, buffer, target, other_meta, curr_it):
        raise NotImplementedError


class TrainMixin:
    def _init(self: TrainProto):
        self.model.train()

    def _normal_step(self: TrainProto, data, target, other_meta, curr_it) -> torch.Tensor:
        """Run one forward/backward/optimizer step and record detailed CUDA timings."""
        # Create CUDA events on the active stream so the async trainer can split model time.
        detail_events = _new_train_detail_events(self.device)
        if detail_events is not None:
            detail_events["start"].record()

        # Clear stale gradients before computing the new loss.
        self.optimizer.zero_grad(set_to_none=True)

        # Run the forward pass and loss under the configured autocast mode.
        with self.amp_scaler.autocast():
            output = self.model(data)
            loss = self.loss_fn(output, target.to(dtype=output.dtype))
        if detail_events is not None:
            detail_events["forward_end"].record()

        # Backpropagate the scaled loss and optionally clip gradients.
        self.amp_scaler.scale(loss).backward()
        if self.grad_clip_norm is not None:
            self.amp_scaler.unscale_(self.optimizer)
            params = [p for g in self.optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)
        if detail_events is not None:
            detail_events["backward_end"].record()

        # Apply the optimizer update and advance AMP scale.
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        if detail_events is not None:
            detail_events["optimizer_end"].record()

        # Step the learning-rate scheduler after the optimizer update.
        self.lr_sched.step()

        # Persist detail events for the async post-step thread.
        self._last_train_detail_events = detail_events

        # Return a detached scalar loss tensor for logging.
        loss2 = loss.detach()
        if loss2.dim() == 0:
            loss2.unsqueeze_(0)
        return loss2

    def _step(self: TrainProto, data, target, other_meta, curr_it) -> torch.Tensor:
        return self._normal_step(data, target, other_meta, curr_it)


    def _pre_step(self, curr_it):
        raise NotImplementedError

    def _post_step(self: TrainProto, res: list, buffer: torch.Tensor, target, other_meta, curr_it):
        raise NotImplementedError


class AsyncInference(InferenceMixin, _AsyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, timer_window_size: int
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=False, timer_window_size=timer_window_size, grad_clip_norm=None
        )


class SyncInference(InferenceMixin, _SyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, timer_window_size: int
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=False, timer_window_size=timer_window_size, grad_clip_norm=None
        )


class AsyncTrainer(TrainMixin, _AsyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        loss_fn: Callable, optimizer: torch.optim.Optimizer,
        lr_sched: Scheduler,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, timer_window_size: int,
        grad_clip_norm: float | None,
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=True, timer_window_size=timer_window_size, grad_clip_norm=grad_clip_norm
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_sched = lr_sched


class SyncTrainer(TrainMixin, _SyncBase):
    def __init__(
        self, model: nn.Module, dataloader: Iterable | torch.utils.data.DataLoader,
        loss_fn: Callable, optimizer: torch.optim.Optimizer,
        lr_sched: Scheduler,
        amp_scaler: MyScaler, profiler: MyProfiler,
        device: torch.device, timer_window_size: int,
        grad_clip_norm: float | None,
    ):
        super().__init__(
            model=model, dataloader=dataloader,
            amp_scaler=amp_scaler, profiler=profiler, device=device,
            move_target=True, timer_window_size=timer_window_size, grad_clip_norm=grad_clip_norm
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_sched = lr_sched


def _new_train_detail_events(device: torch.device) -> dict[str, Event] | None:
    """Create CUDA events used to split one train step into forward/backward/optimizer."""
    # Detailed timing is only meaningful for CUDA training.
    if torch.device(device).type != "cuda":
        return None

    # Allocate events without synchronization; elapsed_time is queried after step completion.
    return {
        "start": torch.cuda.Event(enable_timing=True),
        "forward_end": torch.cuda.Event(enable_timing=True),
        "backward_end": torch.cuda.Event(enable_timing=True),
        "optimizer_end": torch.cuda.Event(enable_timing=True),
    }


def _train_detail_elapsed_ms(events) -> tuple[float, float, float]:
    """Return forward/backward/optimizer elapsed times from one train-step event set."""
    # Non-training paths pass None, so keep their timing fields at zero.
    if events is None:
        return 0.0, 0.0, 0.0

    # Compute adjacent intervals from already-completed CUDA events.
    forward_ms = float(events["start"].elapsed_time(events["forward_end"]))
    backward_ms = float(events["forward_end"].elapsed_time(events["backward_end"]))
    optimizer_ms = float(events["backward_end"].elapsed_time(events["optimizer_end"]))
    return forward_ms, backward_ms, optimizer_ms
