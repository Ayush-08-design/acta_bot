from langchain.tools import tool
from pydantic_monty import Monty, MontyComplete

example_code = """
def calculate_growth(initial, rate, years):
    total = initial
    for _ in range(years):
        total += total * rate
    return round(total, 2)

result = calculate_growth(1000, 0.05, 5)
return result
"""

@tool
async def execute_python_code(llm_code : str):
    '''This tool can execute simple python scripts'''
    # Initialize the secure pool and Open an isolated session
    with Monty() as pool, pool.checkout() as session:
        snapshot = session.feed_start(llm_code)

        while not isinstance(snapshot, MontyComplete):
            snapshot = snapshot.resume_auto()
        return snapshot.output
