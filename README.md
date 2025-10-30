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
### Step 0. Prepare reference images
Before touching the clip, you generate a “character sheet” for the target demographic. <br>
One or several still portraits of the replacement actor are generated with very explicit constraints:
“Generate a Black woman in her early 30s with warm brown skin tone, natural curly hair around shoulder`s length, light makeup, wearing a beige casual jacket similar in silhouette to the original subject’s jacket. Neutral expression, front view.”
“Generate the same woman, same face, same hair, slight smile, 3/4 angle, consistent lighting.”
We save these portraits and treat them as the ground truth identity for that localized version. Think of this as the casting decision for that region.

Why this matters:
This gives us a stable visual identity embedding (face geometry, skin tone, hair texture, clothing silhouette). You are no longer asking the diffusion model to “invent a random Black woman.” You are telling it “use this exact Black woman.”

### Step 1. Shot segmentation
Split the full video into coherent segments (“shots” or “clips”).<br>
**Output:** clip_1.mp4, clip_2.mp4, …

### Step 2. Extract conditioning frames per clip
For each clip, extract the first frame F_start and the last frame F_end.<br>
**Output:** F_start_1.jpg, F_end_1.jpg

### Step 3. Edit demographic attributes on those frames
Send F_start and F_end to a Gemini Flash 2.5 model. When we do this we must edit F_start (first frame of the clip), we do not just prompt “make them Black.”

Our prompt:
“Replace the person with THIS SAME WOMAN from the reference set. Keep pose, body orientation, arm position, lighting, scene layout, and all background elements identical. Do not change camera angle. Do not alter any foreground text.”
* We pass the canonical identity images as reference conditioning.
* The background, props, and framing must remain structurally aligned with the source.

Result:
We get F_start_edited and F_end_edited with the same synthetic actor, not two different actors.

**Output:**
* F_start_edited: first frame with demographically transformed subjects
* F_end_edited: last frame with the same transformed subjects

### Step 4. Video generation from step 3. edited video clips
Send F_start_edited and F_end_edited to Veo3.1 model as first_frame and last_frame arguments.
**Output:** clip_1_edited has consistent people across time, with changed demographic attributes, and approximately the same motion and composition as the original.

### Step 5. Reassemble
Concatenate all regenerated clip_i in the original order and duration to reconstruct the full edited video timeline.
