import requests
import json

BASE_URL = "http://localhost:8000"


def test_rerank(query, prompt, top_n=50, method="cosine"):
    """Test rerank endpoint"""
    payload = {
        "query": query,
        "prompt": prompt,
        "top_n": top_n,
        "method": method
    }

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Prompt: {prompt}")
    print(f"Method: {method}, Top N: {top_n}")
    print(f"{'='*80}\n")

    response = requests.post(f"{BASE_URL}/search_rerank", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['total']} relevant tables:\n")

        for i, result in enumerate(data['results'], 1):
            print(f"{i}. {result['table_name']} (Score: {result['score']}/10)")
            print(f"   Description: {result['table_description'][:150]}...")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    # Test 1: School data
    test_rerank(
        query="школьные данные",
        prompt="Нужны таблицы с информацией о школах, учениках и образовательных учреждениях",
        top_n=20
    )

    # Test 2: Healthcare data
    test_rerank(
        query="медицинские учреждения",
        prompt="Ищу данные о больницах, поликлиниках и системе здравоохранения",
        top_n=15
    )

    # Test 3: Transport data
    test_rerank(
        query="транспортная инфраструктура",
        prompt="Нужны данные о дорогах, общественном транспорте и транспортных потоках",
        top_n=10
    )
