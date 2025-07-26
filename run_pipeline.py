"""
Simple runner script for the AQI Prediction Pipeline.
Execute this script to run the complete pipeline.
"""

from pipeline import main

if __name__ == "__main__":
    print("Starting AQI Prediction Pipeline...")
    print("=" * 50)
    
    try:
        results = main()
        print("\nPipeline execution completed!")
        
        # Show key results
        if 'step_results' in results:
            successful_steps = sum(1 for step in results['step_results'].values() 
                                  if step.get('status', '').startswith('success'))
            total_steps = len(results['step_results'])
            
            if successful_steps == total_steps:
                print("✅ All steps completed successfully!")
            else:
                print(f"⚠️  {successful_steps}/{total_steps} steps completed successfully")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        print("Check pipeline.log for detailed error information.")
