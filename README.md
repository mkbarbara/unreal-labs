# Problem description
The task is to programmatically generate localized ad creatives from a single source video, without reshooting.

### We are given
* An input marketing video in typical UGC style (handheld, lifestyle footage). 
* The video has a static text overlay in the foreground (offer, CTA, discount, etc.). The text layer is large, does not move, and must remain pixel-identical.
* Behind the text, there are people (1–2 subjects, sometimes more) and some background scene (walking, hugging, holding a product, talking, etc.).

### Goal
Generate a new version of that same video where:
* The scene composition, camera motion, lighting, product placement, timing, background, and text overlay do not change.
* The only change is the demographic appearance of the people on screen (for example, replacing white actors with Black actors, or switching gender presentation, or adjusting age range / clothing style for a given market).
* The script must take only the original video as input. There is no manual per-shot rotoscoping, no manual masks, and no human-provided reference frames at inference time.
* The output must be suitable for performance marketing A/B tests across regions.

### Difficulties
* We cannot modify the text layer. It must remain identical, sharp, and readable for compliance and brand safety.
* We must maintain temporal consistency of identity: the “new” Black woman, for example, must look like the same Black woman in every frame of that segment. Otherwise the result looks fake and will not pass review.
* We cannot rely on a proprietary end-to-end video-to-video model like SORA 2 that ingests arbitrary footage and regenerates the same motion with edited people, because this capability is either unavailable or restricted.
* We need to solve identity control, motion consistency, foreground/background separation, and reintegration of the unchanged text layer, all in an automated pipeline.

# Solution
### Step 0. Extract text layers from video
Extract the static text overlay from the original video to preserve it pixel-perfectly for later reintegration.
**Output:** text_rgba.png (transparent PNG with text layer)

### Step 1. Split video into fixed intervals
Split the full video into fixed time intervals (e.g., 5-second segments).
**Output:** video intervals with extracted frames (first and last frame of each interval)

### Step 2. Remove text from extracted frames
Remove text overlays from all extracted frames in video intervals for better performance in subsequent steps.
**Output:** cleaned frames without text overlays

### Step 3. Detect and describe people
Analyze cleaned frames to detect and describe people present in the video (appearance, clothing, position).
**Output:** original_person_registry (descriptions of people in the original video)

### Step 4. Generate reference images
Generate reference images of new people based on the transformation theme (e.g., "Black people", "Asian people").
Create a "character sheet" for the target demographic with explicit constraints to maintain consistent identity across all frames.
**Output:** new_person_registry with reference images for each person

### Step 5. Edit frames with reference images
Edit the cleaned frames by replacing people with the reference images while maintaining pose, body orientation, lighting, scene layout, and background elements.
**Output:** edited frames with demographically transformed subjects

### Step 6. Generate video intervals
Send edited first and last frames of each interval to Veo3.1 model to generate video clips with consistent people across time.
**Output:** generated video intervals with changed demographic attributes

### Step 7. Reassemble video
Concatenate all regenerated video intervals in the original order and duration to reconstruct the full edited video timeline.
**Output:** reassembled_video.mp4

### Step 8. Add text layer
Overlay the extracted text layer from Step 0 back onto the reassembled video to restore the original text overlay.
**Output:** final_video.mp4 (complete localized video with original text preserved)
