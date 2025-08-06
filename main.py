from improved_twinbot import build_langgraph_workflow
from dotenv import load_dotenv
from datetime import datetime

def display_results(result):
    """Display the query results in a formatted way"""
    print("\n=== Query Results ===")
  
    print("\n=== Natural Language Summary ===")
    print(result.get("natural_answer", "No summary available."))

def main():
    # Load environment variables
    load_dotenv()
    
    # Build the workflow
    workflow = build_langgraph_workflow()
    
    print("\n=== Telemetry Data Query Tool ===")
    print("Enter your query about telemetry data (e.g., 'Show me data for the last 7 days')")
    print("Type 'exit' to quit the program")
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() == 'exit':
                print("Exiting program...")
                break
            
            if not query:
                print("Please enter a valid query")
                continue
            
            # Record start time
            start_time = datetime.now()
            
            # Execute the workflow
            result = workflow.invoke({"user_query": query})
            
            # Record end time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Display results
            display_results(result)
            
            print(f"\nQuery executed in {execution_time:.2f} seconds")
            print("="*50)
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again with a different query")

if __name__ == "__main__":
    main()