desc_task = '''
On a precise scale from 0 to 100, rate whether the candidate caption is appropriate for the given image.
'''

desc_vis_info = '''
Use the image and following visual context to evaluate the candidate caption:
'''

desc_answer_format = '''
Your final rating must be a single digit between 0 and 100.
'''

desc_appropriateness = '''
Appropriateness(0-100): How well the caption matches the image's content, atmosphere, and context.
A score of 0 indicates complete inappropriateness, while a score of 100 indicates perfect appropriateness.
'''

desc_precision = '''
Assess the accuracy of the caption by considering the following questions:

Does the caption accurately describe the essential elements, such as objects, people, scene, and situation, depicted in the image?
Is the caption free from inaccurate statements or misinformation?
Does the caption avoid including subjective opinions, speculations, or imaginative interpretations not directly supported by the image?
'''

desc_coverage = '''
Evaluate the comprehensiveness of the caption by addressing the following points:

Does the caption cover all the main elements, objects, and subjects present in the image?
Are the key actions, interactions, or relationships between the elements in the image described?
Does the caption provide relevant context or background information necessary for understanding the image?
Are important details, such as the setting, time, location, or significant visual characteristics, mentioned when applicable?
'''