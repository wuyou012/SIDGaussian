import json
import os
import numpy as np
# Define the input and output file names
# on = '4_1_1'
# input_file = f'./output_{on}/test_results.json'  # The file containing the improperly formatted JSON
# output_file = f'output_{on}/'  # The file to save the corrected JSON
# folder_names = [
#                 "output_final/output_A2D2C2_1", 
#                 "output_final/output_A2D2C2_2",
#                 "output_final/output_A2D2C2_3",
#                 "output_final/output_A2D2C2_4",
#                 "output_final/output_A2D2C2_5",
#                 "output_final/output_A2D2C2_6",
#                 "output_final/output_A2D2C2_7",
#                 "output_final/output_A2D2C2_8",
#                 "output_final/output_A2D2C2_11", 
#                 "output_final/output_A2D2C2_12",
#                 "output_final/output_A2D2C2_13",
#                 "output_final/output_A2D2C2_14",
#                 "output_final/output_A2D2C2_15",
#                 "output_final/output_A2D2C2_16",
#                 "output_final/output_A2D2C2_17",
#                 "output_final/output_A2D2C2_18",
#                 "output_final/output_A2D2C2_21", 
#                 "output_final/output_A2D2C2_22",
#                 "output_final/output_A2D2C2_23",
#                 "output_final/output_A2D2C2_24",
#                 "output_final/output_A2D2C2_25",
#                 "output_final/output_A2D2C2_26",
#                 "output_final/output_A2D2C2_27",
#                 "output_final/output_A2D2C2_28",
#                ]

folder_names = [
                "output_eval", 

               ]
n = 0
def read_improper_json(file_path):
    with open(file_path, 'r') as file:
        # Read the entire file content
        content = file.read()
        
        # Split the content by '}{' to separate individual JSON objects
        json_objects = content.split('}{')
        
        # Add back the braces and create a list of dictionaries
        corrected_objects = []
        for obj in json_objects:
            # Add braces back to the split objects
            if not obj.startswith('{'):
                obj = '{' + obj
            if not obj.endswith('}'):
                obj = obj + '}'
            corrected_objects.append(json.loads(obj))
        
        return corrected_objects

for folder_name in folder_names:
    input_file = f'./{folder_name}/test_results.json'  # The file containing the improperly formatted JSON
    output_file = f'{folder_name}/'  # The file to save the corrected JSON

    corrected_data = read_improper_json(input_file)

    results = []

    # Lists to store highest scores, last scores, and highest score indices for later calculations
    highest_scores_psnr = []
    highest_scores_ssim = []
    highest_scores_lpips = []
    last_PSNR = []
    last_SSIM = []
    last_LPIPS = []
    index_iter = n
    # Process each scene
    for entry in corrected_data:

        scene = entry["scene"]
        PSNR = entry["PSNR"]
        SSIM = entry["SSIM"]
        LPIPS = entry["LPIPS"]
        
        # highest_psnr = max(PSNR)  # Calculate the highest score for the scene
        # highest_ssim = max(SSIM)  # Calculate the highest score for the scene
        # highest_lpips = min(LPIPS)  # Calculate the highest score for the scene
        n_ = n
        highest_psnr_index = np.argmax(PSNR[n_:])+n_
        # if (highest_psnr_index==0):
        #     highest_psnr_index = np.argmax(PSNR[1:])
        highest_psnr = PSNR[highest_psnr_index]
        highest_ssim = SSIM[highest_psnr_index]
        highest_lpips = LPIPS[highest_psnr_index]
        highest_score_index = PSNR.index(highest_psnr)* 1000 + 1000  # Get the index of the highest score
        
        highest_scores_psnr.append(highest_psnr)
        highest_scores_ssim.append(highest_ssim)
        highest_scores_lpips.append(highest_lpips)
        # last_PSNR.append(PSNR[-1])      # Append the last PSNR score
        # last_SSIM.append(SSIM[-1])      # Append the last SSIM score
        # last_LPIPS.append(LPIPS[-1])      # Append the last LPIPS score
        last_PSNR.append(PSNR[index_iter])      # Append the last PSNR score
        last_SSIM.append(SSIM[index_iter])      # Append the last SSIM score
        last_LPIPS.append(LPIPS[index_iter])      # Append the last LPIPS score
        
        # Save detailed results
        results.append({
            "scene": scene,
            "highest_psnr": highest_psnr,
            "highest_ssim": highest_ssim,
            "highest_lpips": highest_lpips,
            "iteration": highest_score_index,
            "last_PSNR": PSNR[index_iter],
            "last_SSIM": SSIM[index_iter],
            "last_LPIPS": LPIPS[index_iter]
        })
        # print("scene", scene, "highest_score", highest_psnr, "iteration", highest_score_index, "last_PSNR", PSNR[-1], "last_SSIM", SSIM[-1], "last_LPIPS", LPIPS[-1])

    average_highest_psnr = sum(highest_scores_psnr) / len(highest_scores_psnr) if highest_scores_psnr else 0
    average_highest_ssim = sum(highest_scores_ssim) / len(highest_scores_ssim) if highest_scores_ssim else 0
    average_highest_lpips = sum(highest_scores_lpips) / len(highest_scores_lpips) if highest_scores_lpips else 0
    average_last_PSNR = sum(last_PSNR) / len(last_PSNR) if last_PSNR else 0
    average_last_SSIM = sum(last_SSIM) / len(last_SSIM) if last_SSIM else 0
    average_last_LPIPS = sum(last_LPIPS) / len(last_LPIPS) if last_LPIPS else 0

    output_data = {
        "overall_averages": {
            "highest_psnr": average_highest_psnr,
            "highest_ssim": average_highest_ssim,
            "highest_lpip": average_highest_lpips,
            "last_PSNR": average_last_PSNR,
            "last_SSIM": average_last_SSIM,
            "last_LPIPS": average_last_LPIPS
        },
        "detailed_results": results
    }
    # print("highest", average_highest_psnr, "last_PSNR", average_last_PSNR, "last_SSIM", average_last_SSIM, "last_LPIPS", average_last_LPIPS)
    # Save the output data to a JSON file
    if(len(corrected_data)!=8):
    # if(0):
        print("Miss scene!!!!!!!!!!!!!!!")
    else:
        with open(os.path.join(output_file, 'overall_averages.json'), 'w') as file:
            json.dump(output_data, file, indent=4)

        # Print a message
        print(f"Data saved to {output_file}/overall_averages.json")