
from src.data.collect_sequences import get_sequences

# Test ENSP ID
test_id = "ENSP00000327694"
print(f"Fetching sequence for {test_id}...")
results = get_sequences([test_id])
print(results)

if not results:
    print("Failed to fetch sequence for ENSP ID.")
