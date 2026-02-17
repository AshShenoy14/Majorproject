
from src.data.sequence_manager import SequenceManager


# Test ENSP ID
test_id = "ENSP00000327694"
print(f"Fetching sequence for {test_id}...")
manager = SequenceManager()
results = manager.get_sequences([test_id])
print(results)

if not results:
    print("Failed to fetch sequence for ENSP ID.")
