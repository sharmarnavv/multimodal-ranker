import sys
import os

# ensure src in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.searcher import CreativeSearch

def run_test():
    searcher = CreativeSearch()

    # scenario:
    # hard: model (must)
    # soft: london (boost)
    
    query = "Cyberpunk neon futuristic"
    
    hard_constraints = {
        "role": "Model"
    }
    
    soft_preferences = {
        "location": "London" 
    }

    print("🚀 running soft search...")
    results = searcher.search(
        text_query=query, 
        hard_filters=hard_constraints,
        soft_filters=soft_preferences
    )

    print(f"\nTop Matches:\n")
    
    for i, hit in enumerate(results):
        payload = hit.payload
        is_boosted = payload.get("_debug_boost")
        
        # formatting
        boost_tag = f"BOOSTED ({is_boosted})" if is_boosted else ""
        print(f"#{i+1} | Score: {hit.score:.4f} {boost_tag}")
        print(f"    Name: {payload.get('name')}")
        print(f"    Location: {payload.get('location')}")
        print("-" * 40)

if __name__ == "__main__":
    run_test()