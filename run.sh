#!/bin/bash

# Define the array of temperature values
temperatures=(0 0.1 0.2 0.3 0.4 0.5)

# Loop over each temperature value
for temperature in "${temperatures[@]}"
do
    # echo "Processing temperature=$temperature"

    # # Generate code with retries in case of network error
    # while true; do
    #     python eval/scripts/gencode_json.py --input-path "eval/data/problems_all.jsonl" --model gpt-4o-mini --output-dir "eval_results/generated_code_${temperature}" --prompt-dir "eval_results/prompt_${temperature}" --temperature $temperature
        
    #     if [ $? -eq 0 ]; then
    #         echo "Successfully generated code for temperature=$temperature"
    #         break
    #     else
    #         echo "Network error encountered. Retrying..."
    #         sleep 5  # Optional: add a delay before retrying
    #     fi
    # done

    # Test the generated code
    python eval/scripts/test_generated_code.py --jsonl-path "eval/data/problems_all.jsonl" --model gpt-4o-mini --code-dir "eval_results/generated_code_${temperature}" --temperature $temperature
    
    if [ $? -eq 0 ]; then
        echo "Successfully tested code for temperature=$temperature"
    else
        echo "Test failed for temperature=$temperature"
        exit 1  # Exit the script if testing fails
    fi
done

echo "All operations completed."
