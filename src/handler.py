# Native
import shutil
import os
import time
import requests
import re
import base64
import json
# Dependencies
import runpod
from requests.adapters import HTTPAdapter, Retry
from requests_toolbelt import MultipartEncoder

sd_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
sd_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                               Functions                                      #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    # Check if the service is ready to receive requests
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)

def run_inference(params):
    config = {
        "baseurl": "http://127.0.0.1:8888",
        "api": {
            ### Query
            "home": ("GET", "/"),
            "ping": ("GET", "/ping"),
            "queryjob": ("GET", "/v1/generation/query-job"),
            "jobqueue": ("GET", "/v1/generation/job-queue"),
            "jobhistory": ("GET", "/v1/generation/job-history"),
            "models": ("GET", "/v1/engines/all-models"),
            "styles": ("GET", "/v1/engines/styles"),
            ### GenerateV1
            "txt2img":  ("POST", "/v1/generation/text-to-image"),
            "upscale-vary": ("POST", "/v1/generation/image-upscale-vary"), #multipart/form-data
            "inpaint-outpaint": ("POST", "/v1/generation/image-inpaint-outpaint"), #multipart/form-data
            "img2img":  ("POST", "/v1/generation/image-prompt"), #multipart/form-data
            "describe": ("POST", "/v1/tools/describe-image"), #multipart/form-data
            ### Default
            "stop": ("POST", "/v1/generation/stop"),
            ### GenerateV2
            "txt2img-ip": ("POST", "/v2/generation/text-to-image-with-ip"),
            "upscale-vary2": ("POST", "/v2/generation/image-upscale-vary"),
            "inpaint-outpaint2": ("POST", "/v2/generation/image-inpaint-outpaint"),
            "img2img2":  ("POST", "/v2/generation/image-prompt"),
        },
        "timeout": 300
    }
    # Check if the api_name in the recieved obj is supported
    api_name = params["api_name"]
    if api_name in config["api"]:
        api_config = config["api"][api_name]
    else:
        raise Exception("Method '%s' not yet implemented" % api_name)

    api_verb = api_config[0]
    api_path = api_config[1]
    response = {}

    # Check for inpaint preset
    if "inpaint_preset" in params:
        result = inpaint_preset(params)
        if result is not True:
            raise ValueError("inpaint_preset task failed: " + result)
        
    # You can send the input_image, input_mask, and cn_images as PNG: binary encoded into base64 string OR as url string
    input_imgs = {'input_image':None, 'input_mask':None, 'cn_img1':None, 'cn_img2':None, 'cn_img3':None, 'cn_img4':None, 'image_prompts':[None,None,None,None], 'image':None}
    
    def process_img(value):
        if re.search(r'https?:\/\/\S+', value) is not None:
            return requests.get(value).content
        elif re.search(r'^[A-Za-z0-9+/]+[=]{0,2}$', value) is not None and value != "None":
            return base64.b64decode(value)
        else:
            return value
        
    for key, value in input_imgs.items():
        if key in params:
            try:
                if key == "image_prompts":
                    for index, prompt in enumerate(params.get("image_prompts", [])):
                        input_imgs["image_prompts"][index] = process_img(prompt["cn_img"])
                else:
                    input_imgs[key] = process_img(params[key])
            except Exception as e:
                error_message = str(e)
                print("Image conversion task failed: ", error_message)
                return {"error": error_message}
    
    ''' ----------------------------
    Send requests to the Fooocus-API
    ---------------------------- '''
    if api_verb == "GET":
        response = sd_session.get(url='%s%s' % (config["baseurl"], api_path), timeout=config["timeout"])

    if api_verb == "POST":
        # If the request should be multipart/form-data, convert the application/json data into it.
        if api_name in ["upscale-vary", "inpaint-outpaint", "img2img", "describe"]:
            try:
                # Replace the original image params with the processed ones
                for key, value in input_imgs.items():
                    if value is not None:
                        if type(value) == list:
                            for i, value in enumerate(input_imgs['image_prompts']):
                                if value is not None:
                                    params['image_prompts'][i]['cn_img'] = (key+'.png', value, 'image/png')
                        else:
                            if isinstance(value, bytes):
                                params[key] = (key+'.png', value, 'image/png')
                            else:
                                params[key] = value
                # Convert
                multipart_data = MultipartEncoder(fields={**params})
                headers = {'Content-Type': multipart_data.content_type}
                response = sd_session.post(
                    url='%s%s' % (config["baseurl"], api_path),
                    data=multipart_data,
                    headers=headers,
                    timeout=config["timeout"])
            except Exception as e:
                error_message = str(e)
                print("multipart/form-data task failed: ", error_message)
                return {"error": error_message}
        else: # If the final request should be application/json. Send the original request data
            # Convert the processed binary image back to url-safe-base64
            for key, value in input_imgs.items():
                if value is not None:
                    if type(value) == list:
                        for i, value in enumerate(input_imgs['image_prompts']):
                            if isinstance(value, bytes):
                                params['image_prompts'][i]["cn_img"] = base64.b64encode(value).decode('utf-8')
                    elif isinstance(value, bytes):
                        params[key] = base64.b64encode(value).decode('utf-8')
            response = sd_session.post(
                url='%s%s' % (config["baseurl"], api_path),
                json=params, 
                timeout=config["timeout"])

    # --- Return the API response to the RunPod ---        
    content_type = response.headers.get('Content-Type', '')
    if 'application/json' in content_type:
        return response.json()
    else:
        return response.text

def preview_stream(jsn, event):
    try:
        job_finished = False
        api_name = event["input"].get("api_name")
        headers = event["input"].get('preview_headers', {})
        if api_name in ["upscale-vary", "inpaint-outpaint", "img2img", "describe"]:
            if headers != {}: headers = json.loads(headers)
        while job_finished is False:
            preview = sd_session.get('http://127.0.0.1:8888/v1/generation/query-job', params={"job_id":jsn["job_id"], "require_step_preview": "true"}).json()
            requests.post(event["input"]["preview_url"], json=preview, headers=headers)
            if(preview["job_stage"] == "SUCCESS" or preview["job_stage"] == "ERROR"):
                job_finished = True
                return
            time.sleep(int(event["input"].get('preview_interval', 1)))
    except Exception as e:
        error_message = str(e)
        print("async preview task failed: ", error_message)
        return {"error": error_message}
    
def clearOutput():
    try:
        print("Clearing outputs...")
        shutil.rmtree('/workspace/outputs/files')
        os.makedirs('/workspace/outputs/files')
        shutil.rmtree('/workspace/repositories/Fooocus/outputs')
        os.makedirs('/workspace/repositories/Fooocus/outputs')
    except Exception as e:
        error_message = str(e)
        print("clear outputs task failed: ", error_message)
        return {"error": error_message}

def inpaint_preset(params):
    option = params.get("inpaint_preset")
    api_name = params.get("api_name")
    advanced_params = params.get("advanced_params", {})
    
    if api_name in ["upscale-vary", "inpaint-outpaint", "img2img", "describe"]: multipart = True
    else: multipart = False
    
    if option == "Improve Detail": advanced_params.update({"inpaint_disable_initial_latent": False, "inpaint_engine": "None", "inpaint_strength": 0.5, "inpaint_respective_field": 0.0})
    elif option in ["Modify Content", "Inpaint or Outpaint"]: advanced_params.update({"inpaint_disable_initial_latent": True, "inpaint_engine": "v2.6", "inpaint_strength": 1.0, "inpaint_respective_field": 0.0})
    else: return "Preset not found. Be sure to use exactly one of: 'Improve Detail', 'Modify Content' or 'Inpaint or Outpaint'"
    
    if multipart:
        advanced_params = {k: str(v).lower() for k, v in advanced_params.items()}
        params["advanced_params"] = json.dumps(advanced_params, indent=None).replace('"', "'")
    else: params["advanced_params"] = advanced_params
    return True

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    ''' This is the handler function that will be called by the serverless. '''
    # Check for clear outputs option (defaults to True, send "clear_output":false in your payload to keep the images stored on the network volume.)
    # Also works on standalone but does not make much sense since the workers are stateless.
    clear_output = event["input"].get("clear_output", True)
    if clear_output is True or clear_output.lower() == "true":
        clearOutput()

    # Main process
    json = run_inference(event["input"])
    
    # Check for async preview streaming (turn on by adding "async_process":true in your generation params and include custom "preview_url":"https://your.app/endpoint")
    if 'preview_url' in event["input"] and 'job_step_preview' in json:
        preview_stream(json, event)
    
    # Return the output that you want to be returned like pre-signed URLs to output artifacts
    return json

if __name__ == "__main__":
    wait_for_service(url='http://127.0.0.1:8888/v1/generation/text-to-image')

    print("Fooocus API Service is ready. Starting RunPod...")

    runpod.serverless.start({"handler": handler})