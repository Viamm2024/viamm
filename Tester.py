from inference_sdk import InferenceHTTPClient

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# DATASET REFERENCE: https://universe.roboflow.com/cobra-mi40f/final-yolov8-annotation/model/1  --> for paper bill
# DATASET REFERENCE: https://universe.roboflow.com/school-cvec2/peso-coins/model/1 --> for coin bill
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# Initialize the inference client for paper bills
CLIENT_BILLS = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZkHJirV5OTz4cA9P0RPG"
)

# Initialize the inference client for coins
CLIENT_COINS = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZkHJirV5OTz4cA9P0RPG"  # Replace with your API key for coin detection
)

# Path to the image
image_url = 'test14.jpg'  # Replace with the path to your image

# Mapping from class labels to numeric values for paper bills
class_mapping_bills = {
    'Twenty': 20,
    'Fifty': 50,
    'One_Hundred': 100,
    'Two_Hundred': 200,
    'Five_Hundred': 500,
    'One_Thousand': 1000
}

# Mapping from class labels to numeric values for coins
class_mapping_coins = {
    '1 peso': 1,
    '5 pesos': 5,
    '10 pesos': 10,
    '20 pesos': 20
}

def infer_and_print_results(client, image_url, model_id, class_mapping, confidence_threshold=0.5):
    try:
        # Perform inference
        result = client.infer(image_url, model_id=model_id)
        
        # Initialize a dictionary to store detected amounts and their counts
        detected_amounts = {}
        total_amount = 0

        # Check if result is empty or contains errors
        if result:
            # Process each prediction
            for prediction in result.get('predictions', []):
                class_label = prediction.get('class', 'unknown')
                confidence = prediction.get('confidence', 0)
                bbox = prediction.get('bbox', [])

                # Only consider predictions with confidence above the threshold
                if confidence >= confidence_threshold:
                    # Map class label to numeric value
                    amount = class_mapping.get(class_label, None)
                    if amount is not None:
                        if amount in detected_amounts:
                            detected_amounts[amount] += 1
                        else:
                            detected_amounts[amount] = 1

            # Calculate total amount
            for amount, count in detected_amounts.items():
                total_amount += amount * count

            # Print results in the desired format
            print("Detected amounts:")
            for amount, count in detected_amounts.items():
                print(f"{count} x {amount}")

            # Print the total amount
            print(f"Total Amount: {total_amount}")

        else:
            print("No result returned or empty response.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

    return total_amount

# Detect paper bills
print("Paper Bills Detection:")
total_bills = infer_and_print_results(CLIENT_BILLS, image_url, "final-yolov8-annotation/1", class_mapping_bills, confidence_threshold=0.5)

# Detect coins
print("\nCoins Detection:")
total_coins = infer_and_print_results(CLIENT_COINS, image_url, "peso-coins/1", class_mapping_coins, confidence_threshold=0.7)

# Print the total amount of both paper bills and coins
total_amount = total_bills + total_coins
print(f"\nTotal Amount of Paper Bills: {total_bills}")
print(f"Total Amount of Coins: {total_coins}")
print(f"\nCombined Total Amount: {total_amount}")