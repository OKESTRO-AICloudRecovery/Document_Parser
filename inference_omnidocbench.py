import os
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool, Pool
import argparse


from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self, 
            ip='localhost',
            port=8000,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=64,
            dpi = 200, 
            output_dir="./output", 
            min_pixels=None,
            max_pixels=None,
            use_hf=False,
            use_direct_vllm=False,
            model_path="rednote-hilab/dots.ocr",
            gpu_memory_utilization=0.95,
            lora_path=None,
        ):
        self.dpi = dpi

        # default args for vllm server
        self.ip = ip
        self.port = port
        self.model_name = model_name
        
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.use_hf = use_hf
        self.use_direct_vllm = use_direct_vllm
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.lora_path = lora_path
        
        if self.use_hf:
            raise ValueError("HF model is not recommended for inference, please use direct vllm model instead")
        elif self.use_direct_vllm:
            self._load_direct_vllm_model()
            print(f"use direct vllm model, num_thread will be set to 1")
        else:
            print(f"use vllm server model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        """ This function is not implemented for omnidocbench, because the hf model is inference slow and memory inefficient.
        Please use direct vllm model instead.
        But, if you want to use hf model, please refer to the dots_ocr/parser.py file.
        """
        raise NotImplementedError("This function is not implemented for omnidocbench, because the hf model is inference slow and memory inefficient.")

    def _load_direct_vllm_model(self):
        from vllm import LLM, SamplingParams
        from vllm.multimodal.utils import encode_image_base64
        
        # VLLM 엔진 직접 로드 (CLI 명령어와 동일한 설정)
        vllm_kwargs = {
            "model": self.model_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        
        # LoRA 어댑터가 지정된 경우 enable_lora만 설정
        if self.lora_path:
            vllm_kwargs["enable_lora"] = True
            print(f"LoRA support enabled. Adapter path: {self.lora_path}")
        
        self.vllm_engine = LLM(**vllm_kwargs)
        print(f"VLLM engine loaded successfully")
        
        self.encode_image_base64 = encode_image_base64

    def _inference_with_hf(self, image, prompt):
        """ This function is not implemented for omnidocbench, because the hf model is inference slow and memory inefficient.
        Please use direct vllm model instead.
        But, if you want to use hf model, please refer to the dots_ocr/parser.py file.
        """
        raise NotImplementedError("This function is not implemented for omnidocbench, because the hf model is inference slow and memory inefficient.")

    def _inference_with_direct_vllm(self, image, prompt):
        from vllm import SamplingParams
        import io
        import base64
        
        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_url = f"data:image/png;base64,{img_base64}"
        
        # 메시지 구성 (기존 VLLM API와 동일한 형식)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
                ],
            }
        ]
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_completion_tokens,
        )
        
        # VLLM 엔진으로 추론 실행
        try:
            if self.lora_path:
                from vllm.lora.request import LoRARequest
                lora_request = LoRARequest("dotsocr_lora", 1, self.lora_path)
                outputs = self.vllm_engine.chat(
                    messages,
                    sampling_params=sampling_params, 
                    lora_request=lora_request,
                    use_tqdm=False
                )
            else:
                # 기본 모델 사용 시
                outputs = self.vllm_engine.chat(
                    messages, 
                    sampling_params=sampling_params, 
                    use_tqdm=False
                )
        except Exception as e:
            print(f"Chat inference failed: {e}")
            return None
        
        # 결과 추출
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text
        else:
            return None

    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt, 
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <+ {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        elif self.use_direct_vllm:
            response = self._inference_with_direct_vllm(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True) # used for clean output or metric of omnidocbench、olmbench 
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    # with open(md_file_path, "w", encoding="utf-8") as md_file:
                    #     md_file.write(md_content)
                    # md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        if self.use_hf or self.use_direct_vllm:
            num_thread = 1
        else:
            num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_directory(
        self,
        input_dir,
        output_dir="",
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False
        ):
        """
        Parse all image and PDF files in a directory
        """
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 지원되는 파일 확장자 목록
        supported_extensions = image_extensions | {'.pdf'}
        
        # 디렉토리 내 모든 파일 찾기
        input_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                _, file_ext = os.path.splitext(file)
                if file_ext.lower() in supported_extensions:
                    input_files.append(file_path)
        
        if not input_files:
            print(f"No supported files found in directory: {input_dir}")
            return []
        
        print(f"Found {len(input_files)} files to process in directory: {input_dir}")
        
        all_results = []
        for file_path in tqdm(input_files, desc="Processing files"):
            try:
                filename, file_ext = os.path.splitext(os.path.basename(file_path))
                save_dir = output_dir # os.path.join(output_dir, filename)
                # os.makedirs(save_dir, exist_ok=True)
                
                if file_ext.lower() == '.pdf':
                    results = self.parse_pdf(file_path, filename, prompt_mode, save_dir)
                elif file_ext.lower() in image_extensions:
                    results = self.parse_image(file_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
                else:
                    continue
                
                # 각 파일의 결과를 개별 jsonl 파일로 저장
                # with open(os.path.join(output_dir, f"{filename}.jsonl"), 'w', encoding="utf-8") as w:
                #     for result in results:
                #         w.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                all_results.extend(results)
                # print(f"Processed: {file_path}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # 전체 결과를 하나의 jsonl 파일로도 저장
        # with open(os.path.join(output_dir, "all_results.jsonl"), 'w', encoding="utf-8") as w:
        #     for result in all_results:
        #         w.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Directory parsing finished. Total {len(all_results)} pages processed. Results saved to {output_dir}")
        return all_results

    def parse_file(self, 
        input_path, 
        output_dir="", 
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False
        ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")
        
        print(f"Parsing finished, results saving to {save_dir}")
        with open(os.path.join(output_dir, os.path.basename(filename)+'.jsonl'), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results



def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input PDF/image file path or directory path"
    )
    
    parser.add_argument(
        "--directory", action='store_true',
        help="Process all files in the input directory"
    )
    
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=16,
        help=""
    )
    parser.add_argument(
        "--no_fitz_preprocess", action='store_true',
        help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--use_hf", type=bool, default=False,
        help=""
    )
    parser.add_argument(
        "--use_direct_vllm", action='store_true',
        help="Use direct VLLM engine instead of API server"
    )
    parser.add_argument(
        "--model_path", type=str, default="rednote-hilab/dots.ocr",
        help="Path to the model for direct VLLM loading"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization for direct VLLM engine"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to LoRA adapter weights (e.g., ./outputs/dotsocr-ft-lora)"
    )
    args = parser.parse_args()

    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=args.use_hf,
        use_direct_vllm=args.use_direct_vllm,
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_path=args.lora_path,
    )

    fitz_preprocess = not args.no_fitz_preprocess
    if fitz_preprocess:
        print(f"Using fitz preprocess for image input, check the change of the image pixels")
    
    if args.directory:
        result = dots_ocr_parser.parse_directory(
            args.input_path,
            prompt_mode=args.prompt,
            bbox=args.bbox,
            fitz_preprocess=fitz_preprocess,
        )
    else:
        result = dots_ocr_parser.parse_file(
            args.input_path, 
            prompt_mode=args.prompt,
            bbox=args.bbox,
            fitz_preprocess=fitz_preprocess,
        )    


if __name__ == "__main__":
    main()