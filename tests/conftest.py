import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_execution():
    device = torch.get_default_device()
    threads = torch.get_num_threads()
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    yield
    torch.set_default_device(device)
    torch.set_num_threads(threads)
