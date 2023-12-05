import os
import ffmpeg
import cv2

def arrange_images_by_name(input_dir):
  """Arranges images in a directory by name.

  Args:
    input_dir: The path to the input directory containing the images.
  """

  # Get a list of all the images in the input directory.
  images = os.listdir(input_dir)

  # Sort the images by name.
  images.sort()

  return images

def ffmpeg_images_to_video(input_dir, output_video):
  """Converts images in a directory to a video.

  Args:
    input_dir: The path to the input directory containing the images.
    output_video: The path to the output video file.
  """

  # Create a ffmpeg process.
  process = ffmpeg.input(f"{input_dir}/%d.jpg", format="image2")

  # Set the output format to video.
  process = process.output(output_video, format="mp4")

  # Run the ffmpeg process.
  process.run()
def cv2_images_to_video(image_folder, video_name,fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x[:-4]))  # Sort images based on filename (assuming they are numbered)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# Example usage:

input_dir = "/mnt/datadrive/annh/metrabs/video0_kps"
output_video = "/mnt/datadrive/annh/metrabs/video0_kps/output_video.mp4"

# Arrange the images in the input directory by name.
images = arrange_images_by_name(input_dir)

# Convert the images to a video.
ffmpeg_images_to_video(input_dir, output_video)