from threading import Thread

def check_works_thread(works, daemon=True, type=None):
    def wait_works(works):
        for work in works:
            work.wait()

    t = Thread(target=wait_works, args=(works,), daemon=daemon)
    t.start()
    return works#t

def check_work_thread(work, daemon=True, type=None):
    def wait_works(work):
        work.wait()
    
    t = Thread(target=wait_works, args=(work,), daemon=daemon)
    t.start()
    return work #t