#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def blockingRun(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def nonBlockingRun(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def allReduceRun(rank, size):
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def blockingExample(size):
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, blockingRun))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def nonBlockingExample(size):
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, nonBlockingRun))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def allReduceExample(size):
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, allReduceRun))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    size = 2
    print('---- blocking test ----')
    blockingExample(size)

    print('---- non blocking test ----')
    nonBlockingExample(size)

    print('---- all reduce test ----')
    allReduceExample(size)
