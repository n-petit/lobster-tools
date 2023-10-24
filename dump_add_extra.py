"""Add just a few tickers that weren't auto done.."""
import subprocess
import socket
from concurrent.futures import ProcessPoolExecutor
from lobster_tools.preprocessing import infer_ticker_dict
import socket
import numpy as np

def process_ticker(ticker, ticker_till_end, full):
    """Chain of mkdir, unzip, write to arctic db, remove tmp folder"""

    tmp_dir = f"/nfs/home/nicolasp/home/data/tmp/{ticker_till_end}"
    raw_data = f"{full}"

    subprocess.run(["mkdir", tmp_dir])
    
    subprocess.run(["7z", "x", raw_data, f"-o{tmp_dir}"])
    
    subprocess.run(["arctic", "--s3", "--library=2021", "single-write", f"--ticker={ticker}"], capture_output=True, text=True)

    # don't rm just now
    # subprocess.run(["rm", "-rf", tmp_dir])

def demo(ticker, ticker_till_end, full):
    """Chain of mkdir, unzip, write to arctic db, remove tmp folder"""
    print(ticker, ticker_till_end, full)

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"starting on hostname {host_name}")
    finfo = infer_ticker_dict("/nfs/lobster_data/lobster_raw/2021")

    tickers_to_add = ['CTRA', 'CVX', 'DVN', 'EQT', 'HAL', 'MPC', 'OXY', 'SLB', 'TRGP', 'XOM']
    # failed with most but failed silently
    # tickers_to_add = ['COP']
    # """Note: CTRVA, EQT, TRGP not there"""
    finfo = [x for x in finfo if x.ticker in tickers_to_add]

    # finfo = [f.ticker for f in finfo]
    # print(finfo)
    servers = ["omi-rapid-" + x for x in ["20", "21"]]
    job_chunks = np.array_split(finfo, len(servers))
    server_to_jobs = {server: job_chunk.tolist() for server, job_chunk in zip(servers, job_chunks)}
    jobs = server_to_jobs[host_name]
    # print(jobs)
    tickers, tickers_till_end, full = zip(*[(f.ticker, f.ticker_till_end, f.full) for f in jobs])

    # print(pformat(server_to_jobs))
    # print(job_chunks)
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        # executor.map(process_ticker, tickers, tickers_till_end, full)
        # demo run
        executor.map(process_ticker, tickers, tickers_till_end, full)