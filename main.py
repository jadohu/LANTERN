import argparse
import sys
import entrypoints.train_drafter.main as train_drafter
import entrypoints.generate_train_data as generate_train_data
import entrypoints.generate_codebook as generate_codebook
import entrypoints.generate_images as generate_images
import entrypoints.extract_code as extract_code
import LANTERN.entrypoints.eval_fid_clip as eval_fid_clip
import entrypoints.eval_prec_recall as eval_prec_recall
import entrypoints.eval_hpsv2 as eval_hpsv2


def get_task_parser(task_name):
    if task_name == "train_drafter":
        return train_drafter.parse_args()
    elif task_name == "generate_train_data":
        return generate_train_data.parse_args()
    elif task_name == "generate_codebook":
        return generate_codebook.parse_args()
    elif task_name == "generate_images":
        return generate_images.parse_args()
    elif task_name == "extract_code":
        return extract_code.parse_args()
    elif task_name == "eval_fid_clip":
        return eval_fid_clip.parse_args()
    elif task_name == "eval_prec_recall":
        return eval_prec_recall.parse_args()
    elif task_name == "eval_hpsv2":
        return eval_hpsv2.parse_args()
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    
def get_task_runner(task_name):
    if task_name == "train_drafter":
        return train_drafter.run_train_drafter
    elif task_name == "generate_train_data":
        return generate_train_data.run_generate_data
    elif task_name == "generate_codebook":
        return generate_codebook.run_generate_codebook
    elif task_name == "generate_images":
        return generate_images.run_generate_image
    elif task_name == "extract_code":
        return extract_code.run_extract_code
    elif task_name == "eval_fid_clip":
        return eval_fid_clip.run_eval_fid_clip
    elif task_name == "eval_prec_recall":
        return eval_prec_recall.run_eval_prec_recall
    elif task_name == "eval_hpsv2":
        return eval_hpsv2.run_eval_hpsv2
    else:
        raise ValueError(f"Invalid task name: {task_name}")

def main():
    parser = argparse.ArgumentParser(description='LANTERN')
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands for different tasks")

    subparsers.add_parser("train_drafter", help="Train the drafter model")
    subparsers.add_parser("generate_train_data", help="Generate training data for drafter")
    subparsers.add_parser("generate_codebook", help="Generate codebook")
    subparsers.add_parser("generate_images", help="Generate images")
    subparsers.add_parser("extract_code", help="Extract code from images")
    subparsers.add_parser("eval_fid_clip", help="Evaluate FID and CLIP")
    subparsers.add_parser("eval_prec_recall", help="Evaluate precision and recall")
    subparsers.add_parser("eval_hpsv2", help="Evaluate HPSv2")

    args, remaining_args = parser.parse_known_args()

    task_parser = get_task_parser(args.command)
    task_args = task_parser.parse_args(remaining_args)

    task_runner = get_task_runner(args.command)
    task_runner(task_args)
    
if __name__ == "__main__":
    main()
