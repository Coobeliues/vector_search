import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

def test_search(query, top_n=5, method="cosine"):
    """Test search endpoint"""
    payload = {
        "query": query,
        "top_n": top_n,
        "method": method
    }

    response = requests.post(f"{BASE_URL}/search", json=payload)
    print(f"Search: '{query}' (method={method}):")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

if __name__ == "__main__":
    # Test health
    test_health()

    # Test different queries
    test_search("школьные данные", top_n=3, method="cosine")
    test_search("транспортные данные", top_n=3, method="cosine")
