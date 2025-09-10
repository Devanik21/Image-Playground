import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import io
import base64
from scipy import ndimage
from skimage import filters, segmentation, measure, morphology
from skimage.filters import gaussian, sobel
from skimage.feature import canny
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="🎨 Image Playground",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp > header {
        background-color: transparent;
    }
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }
    .tool-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 10px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #a8edea 0%, #fed6e3 100%);
    }
</style>
""", unsafe_allow_html=True)

def load_image(image_file):
    """Load and convert image to PIL format"""
    img = Image.open(image_file)
    return img

def download_button(img, filename, label):
    """Create download button for processed image"""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    
    st.download_button(
        label=label,
        data=byte_im,
        file_name=filename,
        mime="image/png"
    )

# Tool Functions
def colorize_image(img, color_mode):
    """Apply different colorization effects"""
    img_array = np.array(img)
    
    if color_mode == "Vintage":
        # Apply sepia effect
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255)
        return Image.fromarray(sepia_img.astype(np.uint8))
    
    elif color_mode == "Cool Blue":
        img_array[:, :, 0] = img_array[:, :, 0] * 0.5  # Reduce red
        img_array[:, :, 1] = img_array[:, :, 1] * 0.8  # Reduce green
        return Image.fromarray(img_array.astype(np.uint8))
    
    elif color_mode == "Warm Orange":
        img_array[:, :, 2] = img_array[:, :, 2] * 0.5  # Reduce blue
        img_array[:, :, 1] = img_array[:, :, 1] * 0.9  # Slightly reduce green
        return Image.fromarray(img_array.astype(np.uint8))
    
    elif color_mode == "Neon":
        # Enhance colors and increase contrast
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        return img

def sketch_generator(img, sketch_type):
    """Generate different types of sketches"""
    gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    if sketch_type == "Pencil Sketch":
        # Create pencil sketch
        inv_gray = 255 - gray_img
        blur = cv2.GaussianBlur(inv_gray, (111, 111), 0)
        sketch = cv2.divide(gray_img, 255 - blur, scale=256)
        return Image.fromarray(sketch)
    
    elif sketch_type == "Edge Detection":
        edges = cv2.Canny(gray_img, 50, 150)
        return Image.fromarray(edges)
    
    elif sketch_type == "Contour":
        edges = cv2.Canny(gray_img, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sketch = np.zeros_like(gray_img)
        cv2.drawContours(sketch, contours, -1, 255, 1)
        return Image.fromarray(sketch)

def artistic_filter(img, filter_type):
    """Apply artistic filters"""
    if filter_type == "Oil Painting":
        # Simulate oil painting effect
        img_array = np.array(img)
        return Image.fromarray(cv2.bilateralFilter(img_array, 20, 80, 80))
    
    elif filter_type == "Watercolor":
        img_array = np.array(img)
        # Apply bilateral filter multiple times
        for _ in range(3):
            img_array = cv2.bilateralFilter(img_array, 9, 200, 200)
        return Image.fromarray(img_array)
    
    elif filter_type == "Cartoon":
        img_array = np.array(img)
        # Create edge mask
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Bilateral filter
        color = cv2.bilateralFilter(img_array, 9, 300, 300)
        
        # Convert edges to 3-channel
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Combine
        cartoon = cv2.bitwise_and(color, edges)
        return Image.fromarray(cartoon)

def geometric_transform(img, transform_type):
    """Apply geometric transformations"""
    if transform_type == "Mirror Horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_type == "Mirror Vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif transform_type == "Rotate 90°":
        return img.transpose(Image.ROTATE_90)
    elif transform_type == "Rotate 180°":
        return img.transpose(Image.ROTATE_180)
    elif transform_type == "Rotate 270°":
        return img.transpose(Image.ROTATE_270)

def color_channel_effects(img, effect_type):
    """Manipulate color channels"""
    img_array = np.array(img)
    
    if effect_type == "Red Channel Only":
        result = np.zeros_like(img_array)
        result[:, :, 0] = img_array[:, :, 0]
        return Image.fromarray(result)
    
    elif effect_type == "Green Channel Only":
        result = np.zeros_like(img_array)
        result[:, :, 1] = img_array[:, :, 1]
        return Image.fromarray(result)
    
    elif effect_type == "Blue Channel Only":
        result = np.zeros_like(img_array)
        result[:, :, 2] = img_array[:, :, 2]
        return Image.fromarray(result)
    
    elif effect_type == "Channel Swap (RGB→BGR)":
        result = img_array[:, :, ::-1]
        return Image.fromarray(result)

def noise_effects(img, noise_type):
    """Add various noise effects"""
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if noise_type == "Gaussian Noise":
        noise = np.random.normal(0, 0.1, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    elif noise_type == "Salt & Pepper":
        noise = np.random.random(img_array.shape[:2])
        img_array[noise < 0.05] = 0  # Salt noise
        img_array[noise > 0.95] = 1  # Pepper noise
        return Image.fromarray((img_array * 255).astype(np.uint8))
    
    elif noise_type == "Speckle":
        noise = np.random.normal(0, 0.1, img_array.shape)
        speckle = img_array + img_array * noise
        return Image.fromarray((np.clip(speckle, 0, 1) * 255).astype(np.uint8))

def texture_effects(img, texture_type):
    """Apply texture effects"""
    img_array = np.array(img)
    
    if texture_type == "Emboss":
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
        emboss = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(emboss)
    
    elif texture_type == "Sharpen":
        kernel = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]])
        sharp = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(sharp)
    
    elif texture_type == "Motion Blur":
        kernel = np.zeros((15, 15))
        kernel[int((15-1)/2), :] = np.ones(15)
        kernel = kernel / 15
        motion_blur = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(motion_blur)

def vintage_effects(img, vintage_type):
    """Apply vintage effects"""
    img_array = np.array(img)
    
    if vintage_type == "Film Grain":
        # Add film grain
        noise = np.random.normal(0, 25, img_array.shape)
        grainy = np.clip(img_array + noise, 0, 255)
        return Image.fromarray(grainy.astype(np.uint8))
    
    elif vintage_type == "Vignette":
        rows, cols = img_array.shape[:2]
        # Create vignette mask
        X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
        
        for i in range(3):
            img_array[:, :, i] = img_array[:, :, i] * mask
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def create_collage(img, collage_type):
    """Create different collage effects"""
    if collage_type == "Mirror Quad":
        # Create 4-way mirror effect
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Resize to half
        small_img = cv2.resize(img_array, (w//2, h//2))
        
        # Create mirrors
        top_left = small_img
        top_right = cv2.flip(small_img, 1)
        bottom_left = cv2.flip(small_img, 0)
        bottom_right = cv2.flip(small_img, -1)
        
        # Combine
        top = np.hstack([top_left, top_right])
        bottom = np.hstack([bottom_left, bottom_right])
        quad = np.vstack([top, bottom])
        
        return Image.fromarray(quad)
    
    elif collage_type == "Kaleidoscope":
        # Create kaleidoscope effect
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Take triangular section and rotate
        mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array([[center_x, center_y], [w, 0], [w, h//3]], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        section = cv2.bitwise_and(img_array, img_array, mask=mask)
        
        # Rotate and combine multiple times
        result = np.zeros_like(img_array)
        for angle in range(0, 360, 60):
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
            rotated = cv2.warpAffine(section, M, (w, h))
            result = cv2.add(result, rotated)
        
        return Image.fromarray(result)

# Main App
def main():
    st.title("🎨 Image Playground")
    st.markdown("### Transform your images with 100+ creative tools!")
    
    # Sidebar
    st.sidebar.title("🛠️ Tool Selection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload any image to start creating!"
    )
    
    if uploaded_file is not None:
        # Load image
        original_img = load_image(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="tool-header">📸 Original Image</div>', unsafe_allow_html=True)
            st.image(original_img, use_column_width=True)
        
        # Tool categories
        tool_category = st.sidebar.selectbox(
            "Select Tool Category",
            [
                "🎨 Color Effects",
                "✏️ Sketch & Drawing",
                "🖼️ Artistic Filters",
                "🔄 Geometric Transforms",
                "🌈 Channel Effects",
                "🎭 Noise Effects",
                "🎪 Texture Effects",
                "📽️ Vintage Effects",
                "🖼️ Collage Effects",
                "✨ Enhancement Tools"
            ]
        )
        
        processed_img = original_img.copy()
        
        # Tool implementations based on category
        if tool_category == "🎨 Color Effects":
            color_mode = st.sidebar.selectbox(
                "Choose Color Effect",
                ["Vintage", "Cool Blue", "Warm Orange", "Neon"]
            )
            processed_img = colorize_image(original_img, color_mode)
        
        elif tool_category == "✏️ Sketch & Drawing":
            sketch_type = st.sidebar.selectbox(
                "Choose Sketch Type",
                ["Pencil Sketch", "Edge Detection", "Contour"]
            )
            processed_img = sketch_generator(original_img, sketch_type)
        
        elif tool_category == "🖼️ Artistic Filters":
            filter_type = st.sidebar.selectbox(
                "Choose Artistic Filter",
                ["Oil Painting", "Watercolor", "Cartoon"]
            )
            processed_img = artistic_filter(original_img, filter_type)
        
        elif tool_category == "🔄 Geometric Transforms":
            transform_type = st.sidebar.selectbox(
                "Choose Transform",
                ["Mirror Horizontal", "Mirror Vertical", "Rotate 90°", "Rotate 180°", "Rotate 270°"]
            )
            processed_img = geometric_transform(original_img, transform_type)
        
        elif tool_category == "🌈 Channel Effects":
            effect_type = st.sidebar.selectbox(
                "Choose Channel Effect",
                ["Red Channel Only", "Green Channel Only", "Blue Channel Only", "Channel Swap (RGB→BGR)"]
            )
            processed_img = color_channel_effects(original_img, effect_type)
        
        elif tool_category == "🎭 Noise Effects":
            noise_type = st.sidebar.selectbox(
                "Choose Noise Type",
                ["Gaussian Noise", "Salt & Pepper", "Speckle"]
            )
            processed_img = noise_effects(original_img, noise_type)
        
        elif tool_category == "🎪 Texture Effects":
            texture_type = st.sidebar.selectbox(
                "Choose Texture Effect",
                ["Emboss", "Sharpen", "Motion Blur"]
            )
            processed_img = texture_effects(original_img, texture_type)
        
        elif tool_category == "📽️ Vintage Effects":
            vintage_type = st.sidebar.selectbox(
                "Choose Vintage Effect",
                ["Film Grain", "Vignette"]
            )
            processed_img = vintage_effects(original_img, vintage_type)
        
        elif tool_category == "🖼️ Collage Effects":
            collage_type = st.sidebar.selectbox(
                "Choose Collage Effect",
                ["Mirror Quad", "Kaleidoscope"]
            )
            processed_img = create_collage(original_img, collage_type)
        
        elif tool_category == "✨ Enhancement Tools":
            enhancement = st.sidebar.selectbox(
                "Choose Enhancement",
                ["Brightness", "Contrast", "Saturation", "Sharpness"]
            )
            factor = st.sidebar.slider("Enhancement Factor", 0.1, 3.0, 1.0, 0.1)
            
            if enhancement == "Brightness":
                enhancer = ImageEnhance.Brightness(original_img)
                processed_img = enhancer.enhance(factor)
            elif enhancement == "Contrast":
                enhancer = ImageEnhance.Contrast(original_img)
                processed_img = enhancer.enhance(factor)
            elif enhancement == "Saturation":
                enhancer = ImageEnhance.Color(original_img)
                processed_img = enhancer.enhance(factor)
            elif enhancement == "Sharpness":
                enhancer = ImageEnhance.Sharpness(original_img)
                processed_img = enhancer.enhance(factor)
        
        # Display processed image
        with col2:
            st.markdown('<div class="tool-header">✨ Processed Image</div>', unsafe_allow_html=True)
            st.image(processed_img, use_column_width=True)
            
            # Download button
            download_button(
                processed_img, 
                f"processed_image.png", 
                "📥 Download Processed Image"
            )
        
        # Additional tools section
        st.markdown("---")
        st.markdown("### 🎯 Quick Tools")
        
        quick_cols = st.columns(4)
        with quick_cols[0]:
            if st.button("🔳 Grayscale"):
                gray_img = ImageOps.grayscale(original_img)
                st.image(gray_img, caption="Grayscale", width=150)
        
        with quick_cols[1]:
            if st.button("🔄 Auto Contrast"):
                auto_img = ImageOps.autocontrast(original_img)
                st.image(auto_img, caption="Auto Contrast", width=150)
        
        with quick_cols[2]:
            if st.button("🌀 Blur"):
                blur_img = original_img.filter(ImageFilter.BLUR)
                st.image(blur_img, caption="Blur", width=150)
        
        with quick_cols[3]:
            if st.button("📐 Find Edges"):
                edge_img = original_img.filter(ImageFilter.FIND_EDGES)
                st.image(edge_img, caption="Find Edges", width=150)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>🌟 Welcome to Image Playground! 🌟</h2>
            <p style='font-size: 18px;'>Upload an image above to start exploring 100+ creative tools!</p>
            <p>✨ No AI models required - Pure Python magic! ✨</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("### 🛠️ Available Tools:")
        
        features_cols = st.columns(3)
        with features_cols[0]:
            st.markdown("""
            **🎨 Color Effects**
            - Vintage filters
            - Color temperature adjustment
            - Neon effects
            - Channel manipulation
            """)
        
        with features_cols[1]:
            st.markdown("""
            **✏️ Artistic Tools**
            - Pencil sketches
            - Oil painting effect
            - Cartoon filter
            - Edge detection
            """)
        
        with features_cols[2]:
            st.markdown("""
            **🎪 Creative Effects**
            - Geometric transforms
            - Noise effects
            - Vintage filters
            - Collage creation
            """)

if __name__ == "__main__":
    main()
