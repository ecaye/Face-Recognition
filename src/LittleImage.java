
import java.awt.image.BufferedImage;

public class LittleImage {
	public static final int LITTLE_IMAGE_SIZE = 128;
	private static final int NB_RGB_DIMENSIONS = 3;
	
	private int[] pixels;
	private Label label;
	
	public LittleImage() {
		pixels = new int[LITTLE_IMAGE_SIZE * LITTLE_IMAGE_SIZE * NB_RGB_DIMENSIONS];
	}
	
	public LittleImage(int[] pixels, Label label) {
		this.pixels = pixels;
		this.label = label;
	}

	public int[] getPixels() {
		return pixels;
	}

	public Label getLabel() {
		return label;
	}
	
	public void fromBmp(BufferedImage im){
		int imageWidth = im.getWidth();
		int imageHeight = im.getHeight();

		int pixelIndex = 0;
		
		for (int i = 0; i < imageWidth; i++) {
			for (int j = 0; j < imageHeight; j++) {
				int pixel = im.getRGB(i, j);
				int [] RGBpixels = KNN.pixelToRgbArray(pixel);
				
				pixels[pixelIndex++] = RGBpixels[0];
				pixels[pixelIndex++] = RGBpixels[1];
				pixels[pixelIndex++] = RGBpixels[2];	
			}
		}	
	}
}
