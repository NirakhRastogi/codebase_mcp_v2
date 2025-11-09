from config_loader import load_default_config
from query_pipeline.query_processor import QueryProcessor

if __name__ == "__main__":
    config = load_default_config()
    processor = QueryProcessor(config)

    result = processor.process_query("react")
    print(result["intent"])
    print(result["context"])