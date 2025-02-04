import concurrent.futures
import time

def worker(n):
    print(f'Worker {n} is starting')
    time.sleep(2)
    print(f'Worker {n} is done')
    return n

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f'Result: {result}')

if __name__ == '__main__':
    main()