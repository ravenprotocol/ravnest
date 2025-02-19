from threading import Thread
import torch

def check_works_thread(works, stream=None, daemon=True, type=None):
    def wait_works(works):
        if stream is not None:
            with torch.cuda.stream(stream):
                for work in works:
                    work.wait()
        else:
            for work in works:
                work.wait()

    t = Thread(target=wait_works, args=(works,), daemon=daemon)
    t.start()
    return works#t

def check_work_thread(work, stream=None, daemon=True, type=None):
    def wait_works(work):
        if stream is not None:
            with torch.cuda.stream(stream):
                work.wait()
        else:
            work.wait()
    
    t = Thread(target=wait_works, args=(work,), daemon=daemon)
    t.start()
    return work #t