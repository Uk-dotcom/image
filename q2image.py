import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox

def select_image():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

def process_image(image_path):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read the image file")

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # I. Inverse Transformation
        inverse_img = cv2.bitwise_not(img_rgb)
        
        # II. Contrast Stretching
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        contrast_img = cv2.merge((cl,a,b))
        contrast_img = cv2.cvtColor(contrast_img, cv2.COLOR_LAB2RGB)
        
        # III. Histogram Equalization
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(gray_img)
        equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
        
        # IV. Edge Detection (Canny)
        edges = cv2.Canny(gray_img, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(inverse_img)
        plt.title('Inverse Transformation')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(contrast_img)
        plt.title('Contrast Stretched')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(equalized_img)
        plt.title('Histogram Equalized')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(edges)
        plt.title('Edge Detection (Canny)')
        plt.axis('off')

        plt.tight_layout()
        
        # Save results
        output_folder = os.path.join(os.path.dirname(image_path), 'processed')
        os.makedirs(output_folder, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_folder, '1_inverse.jpg'), cv2.cvtColor(inverse_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, '2_contrast.jpg'), cv2.cvtColor(contrast_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, '3_equalized.jpg'), cv2.cvtColor(equalized_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, '4_edges.jpg'), cv2.cvtColor(edges, cv2.COLOR_RGB2BGR))
        plt.savefig(os.path.join(output_folder, 'combined_results.png'), bbox_inches='tight', dpi=300)
        
        print(f"Results saved to: {output_folder}")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

if __name__ == "__main__":
    import os
    try:
        print("Advanced Image Processing Tool")
        image_path = select_image()
        if image_path:
            process_image(image_path)
    except ImportError as e:
        messagebox.showerror("Missing Dependency", f"Please install:\n\npip install {' '.join(e.name.split(' '))}")
    except Exception as e:
        messagebox.showerror("Error", f"Critical error:\n{str(e)}")