import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini (Dùng cho Phân tích tự động) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# ----------------------------------------------------------------------
#                         PHẦN THÊM CHỨC NĂNG CHAT
# ----------------------------------------------------------------------

# --- Khởi tạo State Chat & Hàm Chat API ---

# Khởi tạo lịch sử chat trong Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Lưu trữ DataFrame đã xử lý để dùng trong Chat
if "df_processed_for_chat" not in st.session_state:
    st.session_state["df_processed_for_chat"] = None

def get_chat_response(prompt, api_key, df_processed):
    """Xử lý logic chat, duy trì lịch sử và gọi Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        # Thêm ngữ cảnh về dữ liệu tài chính vào tin nhắn đầu tiên của cuộc trò chuyện.
        # Hoặc mỗi lần gọi API nếu cần đảm bảo mô hình luôn có ngữ cảnh.
        context_data = df_processed.to_markdown(index=False)
        
        # Thêm hệ thống prompt (role-playing) và ngữ cảnh vào tin nhắn đầu tiên
        # để Gemini biết nó đang nói về dữ liệu nào.
        # Sử dụng API chat.send_message nếu muốn duy trì lịch sử trên model
        # hoặc gán lại toàn bộ lịch sử trong mỗi lần gọi, như cách dưới đây.
        
        # Xây dựng lịch sử tin nhắn cho API, bao gồm cả System Instruction (vai trò)
        # và dữ liệu bối cảnh.
        system_instruction = (
            "Bạn là một chuyên gia phân tích tài chính Python/Streamlit rất giàu kinh nghiệm."
            "Hãy trả lời các câu hỏi của người dùng dựa trên Dữ liệu Tài chính đã cung cấp."
            "Chỉ sử dụng dữ liệu từ bảng để trả lời. Nếu không thể tính toán hoặc không có dữ liệu,"
            "hãy nói rằng bạn không tìm thấy thông tin cần thiết."
        )
        
        # Thêm dữ liệu tài chính vào tin nhắn đầu tiên của user (hoặc system) để làm bối cảnh
        full_prompt = (
            f"{system_instruction}\n\n"
            f"DỮ LIỆU TÀI CHÍNH ĐÃ PHÂN TÍCH:\n{context_data}\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG: {prompt}"
        )
        
        # Tải lại lịch sử chat để gửi lên model
        messages_history = [
            {"role": "user", "parts": [{"text": full_prompt}]}
            if msg["role"] == "user" and i == 0 else # Gán full_prompt vào tin nhắn user đầu tiên
            {"role": "user", "parts": [{"text": msg["content"]}]}
            if msg["role"] == "user" else 
            {"role": "model", "parts": [{"text": msg["content"]}]}
            for i, msg in enumerate(st.session_state.messages)
        ]
        
        # Thêm tin nhắn user hiện tại (chỉ là prompt nếu không phải là tin nhắn đầu)
        if not st.session_state.messages:
             messages_history = [{"role": "user", "parts": [{"text": full_prompt}]}]
        else:
             messages_history.append({"role": "user", "parts": [{"text": prompt}]})
        
        # Gọi API (Có thể thay bằng client.chats.create() và chat.send_message() nếu muốn)
        response = client.models.generate_content(
            model=model_name,
            contents=messages_history
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"
        

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())
        
        # Lưu DataFrame đã xử lý vào Session State để dùng trong Chat
        st.session_state["df_processed_for_chat"] = df_processed

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn 
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                thanh_toan_hien_hanh_N_1 = "N/A"
                
            # --- Chức năng 5: Nhận xét AI (Giữ nguyên) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if 'TÀI SẢN NGẮN HẠN' in df_processed['Chỉ tiêu'].str.upper().str.strip().tolist() else 'N/A', 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # ----------------------------------------------------------------------
            #                         KHUNG CHAT ĐÃ THÊM
            # ----------------------------------------------------------------------
            
            st.divider()
            st.subheader("6. Chat với Gemini AI về Dữ liệu (Hỏi Đáp) 💬")
            
            # Hiển thị các tin nhắn cũ
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Khung nhập liệu mới
            if prompt := st.chat_input("Hỏi Gemini về các chỉ số tài chính (VD: 'Tốc độ tăng trưởng của Tổng tài sản là bao nhiêu?'):"):
                
                api_key = st.secrets.get("GEMINI_API_KEY")
                
                if not api_key:
                    st.error("Lỗi: Không tìm thấy Khóa API. Không thể bắt đầu Chat.")
                    # Thêm tin nhắn người dùng (cho lịch sử, dù không gọi API thành công)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Tin nhắn lỗi của model
                    st.session_state.messages.append({"role": "assistant", "content": "Lỗi API Key. Vui lòng kiểm tra cấu hình Secrets."})
                    with st.chat_message("assistant"):
                        st.markdown("Lỗi API Key. Vui lòng kiểm tra cấu hình Secrets.")
                    return # Dừng hàm
                
                # 1. Thêm tin nhắn người dùng vào lịch sử và hiển thị
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 2. Gọi API để lấy phản hồi
                with st.chat_message("assistant"):
                    with st.spinner("Gemini đang phân tích và trả lời..."):
                        # Gọi hàm mới để xử lý chat
                        full_response = get_chat_response(
                            prompt, 
                            api_key, 
                            st.session_state["df_processed_for_chat"] # Truyền DataFrame đã xử lý
                        )
                    st.markdown(full_response)
                
                # 3. Thêm phản hồi của AI vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # ----------------------------------------------------------------------
            
    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
