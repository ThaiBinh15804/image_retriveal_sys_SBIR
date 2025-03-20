import pandas as pd

def process_comments_chunk(chunk):
    """ Xử lý một phần dữ liệu từ file bình luận """
    data = {}
    for _, row in chunk.fillna("").iterrows():
        image_name = str(row['image_name']).strip()
        comment = str(row['comment']).strip()
        
        if image_name and image_name not in ["File Name"]:  # Lọc các giá trị không hợp lệ
            if image_name in data:
                data[image_name]["comments"].append(comment)
            else:
                data[image_name] = {"comments": [comment], "link": None}
    
    return data

def read_comments_csv(file_path, chunksize=10000):
    """ Đọc CSV bình luận """
    try:
        df_iterator = pd.read_csv(
            file_path, sep=r'\|', names=['image_name', 'comment_number', 'comment'], 
            dtype=str, chunksize=chunksize, encoding='latin-1', on_bad_lines='skip', engine='python'
        )
    except UnicodeDecodeError:
        df_iterator = pd.read_csv(
            file_path, sep=r'\|', names=['image_name', 'comment_number', 'comment'], 
            dtype=str, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip', engine='python'
        )
    
    results = []
    for chunk in df_iterator:
        results.append(process_comments_chunk(chunk))
    
    return merge_comment_dicts(results)

def merge_comment_dicts(comment_dicts):
    """ Hợp nhất dữ liệu từ nhiều chunks comments """
    merged = {}
    for d in comment_dicts:
        for key, value in d.items():
            if key in merged:
                merged[key]["comments"].extend(value["comments"])
            else:
                merged[key] = value
    return merged

def read_links_csv(file_path):
    """ Đọc file chứa link ảnh """
    try:
        df = pd.read_csv(
            file_path, sep=r'\,', names=['image_name', 'image_link'], dtype=str, engine='python'
        )
        return {row['image_name'].strip(): row['image_link'].strip() for _, row in df.fillna("").iterrows()}
    except Exception as e:
        print(f"Lỗi khi đọc file links: {e}")
        return {}

def merge_dicts(comments_data, links_data):
    """ Hợp nhất dữ liệu từ hai file """
    merged = comments_data.copy()
    for image_name, link in links_data.items():
        if image_name in merged:
            merged[image_name]["link"] = link
        else:
            merged[image_name] = {"comments": [], "link": link}
    return merged

if __name__ == "__main__":
    comments_file = "D:/Learn/InSchool/NCKH/image_retriveal_sys/Process/data/output.csv"
    links_file = "D:/Learn/InSchool/NCKH/image_retriveal_sys/Process/data/image_links.csv"

    comments_data = read_comments_csv(comments_file)
    links_data = read_links_csv(links_file)
    unified_data = merge_dicts(comments_data, links_data)

    print(unified_data)