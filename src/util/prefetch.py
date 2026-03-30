import torch


def _record_stream(batch, stream):
    if torch.is_tensor(batch):
        batch.record_stream(stream)
        return
    if isinstance(batch, (list, tuple)):
        for item in batch:
            _record_stream(item, stream)
        return
    if isinstance(batch, dict):
        for item in batch.values():
            _record_stream(item, stream)


class CUDAPrefetcher:
    """Prefetch the next batch to GPU on a dedicated CUDA stream."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.enabled = torch.cuda.is_available() and device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return

        loader_iter = iter(self.loader)
        next_batch = self._preload(loader_iter)

        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = next_batch
            _record_stream(batch, torch.cuda.current_stream(self.device))
            next_batch = self._preload(loader_iter)
            yield batch

    def _preload(self, loader_iter):
        try:
            x, labels = next(loader_iter)
        except StopIteration:
            return None

        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True).to(torch.float32).div_(255)
            x = x * 2.0 - 1.0
            labels = labels.to(self.device, non_blocking=True)

        return x, labels
