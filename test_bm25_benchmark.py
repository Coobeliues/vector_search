"""
Benchmark тест для /search_hybrid_bm25 эндпойнта
Тестирует гибридный поиск с BM25 (vector + BM25 keyword search)
"""
import json
import requests
from typing import Dict, List
from collections import defaultdict


# Настройки API
API_URL = "http://localhost:8000/search_hybrid_bm25"
BENCHMARKS_FILE = "/home/keosido/Desktop/Vector_search/benchmarks.json"


def load_benchmarks() -> List[Dict]:
    """Загрузить тесты из benchmarks.json"""
    with open(BENCHMARKS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_single_query(question: str, target_table: str, top_n: int = 50) -> Dict:
    """
    Отправить запрос к /search_hybrid_bm25 и проверить позицию целевой таблицы

    Args:
        question: Вопрос пользователя
        target_table: Ожидаемая таблица (должна быть в топе)
        top_n: Сколько таблиц запрашивать

    Returns:
        Dict с результатами теста
    """
    # Запрос к API
    payload = {
        "query": question,
        "top_n": top_n,
        "vector_weight": 0.5,  # Равные веса для начала
        "tags_weight": 0.5,    # BM25 weight
        "method": "dot_product",
        "rrf_k": 60
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        return {
            "question": question,
            "target_table": target_table,
            "status": "ERROR",
            "error": str(e),
            "position": None,
            "rrf_score": None,
            "vector_rank": None,
            "bm25_rank": None,
            "total_results": 0
        }

    # Найти позицию целевой таблицы
    position = None
    rrf_score = None
    vector_rank = None
    bm25_rank = None
    total_results = len(results.get("results", []))

    for idx, result in enumerate(results.get("results", []), start=1):
        if result["table_name"] == target_table:
            position = idx
            rrf_score = result["rrf_score"]
            vector_rank = result.get("vector_rank")
            bm25_rank = result.get("tags_rank")  # tags_rank содержит BM25 rank
            break

    # Определить статус
    if position is None:
        status = "NOT_FOUND"
    elif position == 1:
        status = "TOP_1"
    elif position <= 3:
        status = "TOP_3"
    elif position <= 5:
        status = "TOP_5"
    elif position <= 10:
        status = "TOP_10"
    else:
        status = f"RANK_{position}"

    return {
        "question": question,
        "target_table": target_table,
        "status": status,
        "position": position,
        "rrf_score": rrf_score,
        "vector_rank": vector_rank,
        "bm25_rank": bm25_rank,
        "total_results": total_results,
        "top_3_tables": [r["table_name"] for r in results.get("results", [])[:3]]
    }


def run_all_benchmarks(top_n: int = 50) -> Dict:
    """
    Запустить все тесты из benchmarks.json

    Args:
        top_n: Количество таблиц для запроса

    Returns:
        Словарь со статистикой и детальными результатами
    """
    benchmarks = load_benchmarks()
    results = []
    stats = defaultdict(int)

    print(f"Запуск {len(benchmarks)} тестов для /search_hybrid_bm25...")
    print("Параметры: vector_weight=0.5, bm25_weight=0.5, method=dot_product")
    print("=" * 80)

    for idx, benchmark in enumerate(benchmarks, start=1):
        question = benchmark["question"]
        target_table = benchmark["target_table"]

        print(f"\n[{idx}/{len(benchmarks)}] Тест: {question[:60]}...")

        result = test_single_query(question, target_table, top_n)
        results.append(result)

        # Обновить статистику
        stats[result["status"]] += 1

        # Вывести результат
        if result["position"] is None:
            print(f"  ❌ NOT_FOUND - таблица '{target_table}' не найдена!")
        elif result["position"] == 1:
            print(
                f"  ✅ TOP_1 - '{target_table}' на 1 месте! "
                f"(rrf: {result['rrf_score']:.4f}, vec: #{result['vector_rank']}, bm25: #{result['bm25_rank']})"
            )
        else:
            print(
                f"  ⚠️  RANK_{result['position']} - '{target_table}' на {result['position']} месте "
                f"(rrf: {result['rrf_score']:.4f}, vec: #{result['vector_rank']}, bm25: #{result['bm25_rank']})"
            )
            print(f"     Топ-3: {', '.join(result['top_3_tables'])}")

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 80)

    total = len(benchmarks)
    print(f"Всего тестов: {total}")
    print(f"  ✅ TOP_1:  {stats['TOP_1']:3d} ({stats['TOP_1']/total*100:5.1f}%)")
    print(f"  ✅ TOP_3:  {stats['TOP_3']:3d} ({stats['TOP_3']/total*100:5.1f}%)")
    print(f"  ⚠️  TOP_5:  {stats['TOP_5']:3d} ({stats['TOP_5']/total*100:5.1f}%)")
    print(f"  ⚠️  TOP_10: {stats['TOP_10']:3d} ({stats['TOP_10']/total*100:5.1f}%)")

    # Показать все остальные ранги
    other_ranks = {k: v for k, v in stats.items() if k.startswith('RANK_')}
    if other_ranks:
        print(f"  ⚠️  Другие позиции:")
        for rank, count in sorted(other_ranks.items()):
            print(f"     {rank}: {count}")

    print(f"  ❌ NOT_FOUND: {stats['NOT_FOUND']:3d} ({stats['NOT_FOUND']/total*100:5.1f}%)")
    print(f"  ⚠️  ERRORS:    {stats['ERROR']:3d}")

    # Средняя позиция (только для найденных)
    positions = [r["position"] for r in results if r["position"] is not None]
    if positions:
        avg_position = sum(positions) / len(positions)
        print(f"\nСредняя позиция целевой таблицы: {avg_position:.2f}")

    print("=" * 80)

    return {
        "total_tests": total,
        "stats": dict(stats),
        "results": results,
        "avg_position": avg_position if positions else None
    }


def show_failed_tests(results: List[Dict], threshold_position: int = 5):
    """
    Показать тесты, где целевая таблица не в топ-N

    Args:
        results: Результаты всех тестов
        threshold_position: Порог позиции (по умолчанию 5)
    """
    print(f"\n{'=' * 80}")
    print(f"ПРОБЛЕМНЫЕ ТЕСТЫ (целевая таблица не в топ-{threshold_position}):")
    print("=" * 80)

    failed = [
        r for r in results
        if r["position"] is None or r["position"] > threshold_position
    ]

    if not failed:
        print(f"✅ Все таблицы в топ-{threshold_position}!")
        return

    for idx, result in enumerate(failed, start=1):
        print(f"\n[{idx}] Вопрос: {result['question']}")
        print(f"    Целевая таблица: {result['target_table']}")
        if result.get('status') == 'ERROR':
            print(f"    ❌ Ошибка: {result.get('error', 'Unknown error')}")
        elif result['position'] is None:
            print(f"    ❌ Статус: NOT_FOUND")
        else:
            print(
                f"    ⚠️  Статус: позиция {result['position']} "
                f"(rrf: {result['rrf_score']:.4f}, vec: #{result['vector_rank']}, bm25: #{result['bm25_rank']})"
            )
        if 'top_3_tables' in result:
            print(f"    Топ-3 результата: {', '.join(result['top_3_tables'])}")


if __name__ == "__main__":
    # Запустить все тесты
    benchmark_results = run_all_benchmarks(top_n=50)

    # Показать проблемные кейсы (где таблица не в топ-5)
    show_failed_tests(benchmark_results["results"], threshold_position=5)

    # Сохранить детальные результаты
    output_file = "/home/keosido/Desktop/Vector_search/bm25_benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Детальные результаты сохранены в: {output_file}")
