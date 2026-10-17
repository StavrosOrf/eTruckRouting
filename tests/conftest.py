"""Keep small neural-network tests from oversubscribing CPU threads."""

_original_torch_threads = None


def pytest_sessionstart(session):
    global _original_torch_threads
    import torch

    _original_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def pytest_sessionfinish(session, exitstatus):
    if _original_torch_threads is not None:
        import torch

        torch.set_num_threads(_original_torch_threads)
