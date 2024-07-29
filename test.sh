model_name=$1
python eval/scripts/gencode_json.py --model $model_name
python eval/scripts/test_generated_code.py --model $model_name