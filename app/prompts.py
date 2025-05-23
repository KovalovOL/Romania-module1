create_novel_ua = f"""  
**Завдання:**  
Створити коротку повчальну історію для дітей (до 120 слів).  

**Вимоги:**  
- **Сюжет:** Проста ситуація з соціальною взаємодією (наприклад: поділитися іграшкою, допомогти другові).  
- **Емоція:** Вибрати **одну** з цих: радість, здивування, сум, злість, сором. Використовувати лише ці варіанти.  
- **Акцент:** Чітко показати обрану емоцію через дії/діалоги персонажа (наприклад: "Оля засміялася й заплескала в долоні" → радість).  

**Додатково:**  
- Після тексту **виділити головну емоцію** зі списку.  
- Створити **питання з вибором** (4 варіанти, перший – правильний) та **коротке пояснення** (до 16 слів).  
- Написати **опис для ілюстрації** (без стилю, лише сцена: "Хлопчик з собакою в парку, собака несподівано знаходить м’яч під кущем").  
- Не використовувати ніяке форматування тексту 


**Зразок відповіді:**  
Текст:  
"Сашко грав у саду, коли побачив, що його улюблений м’яч зник. Він оббіг усі кущі, але нічого не знайшов. Раптом його пес Барсік почав гавкати під яблунею. Сашко підбіг і побачив м’яч, застряглий між гілками. Він витягнув його, обійняв Барсіка й засміявся: «Ти найкращий детектив!»"  
Емоція: здивування  
Питання: Яку емоцію відчув Сашко, коли знайшов м’яч?  
1. Здивування  
2. Сором  
3. Сум  
4. Злість  
Пояснення: Він не очікував, що пес допоможе, і зрадів.  
Ілюстрація: Хлопчик у саду тримає м’яч, пес гавкає на яблуню, гілки розхитані.  
"""  

create_novel_eng = f"""  
Task:  
Create a short instructive story for children (up to 120 words).  

Requirements:  
- **Plot:** A simple social interaction scenario (e.g., sharing a toy, helping a friend).  
- **Emotion:** Choose **one** from: joy, surprise, sadness, anger, shame. Use only these options.  
- **Focus:** Clearly depict the chosen emotion through the character’s actions/dialogue (e.g., "Olya laughed and clapped her hands" → joy).  

Additional:  
- After the text, **highlight the main emotion** from the list.  
- Create a **multiple-choice question** (4 options, first is correct) and a **short explanation** (up to 16 words).  
- Write an **illustration description** (no style, just the scene: "A boy with a dog in a park, the dog unexpectedly finds a ball under a bush").  
- Do not use any text formatting 

Sample Response:  
Text:  
"Sashko was playing in the garden when he noticed his favorite ball was missing. He searched all the bushes but found nothing. Suddenly, his dog Barsik started barking under the apple tree. Sashko ran over and saw the ball stuck between the branches. He pulled it out, hugged Barsik, and laughed: 'You’re the best detective!'"  
Emotion: Surprise  
Question: What emotion did Sashko feel when he found the ball?  
1. Surprise  
2. Shame  
3. Sadness  
4. Anger  
Explanation: He didn’t expect the dog to help and felt relieved.  
Illustration: A boy in a garden holding a ball, dog barking at an apple tree, branches shaking.  
"""  

def get_image_promt(your_description: str):
    return f"""
Vintage children's book illustration style: soft watercolor textures, pastel tones,
round cute characters with hand-drawn outlines. Stylized background with light layering
(trees, clouds). Style reference: Eric Carle, Soviet fairy tales from the 1970s.
No 3D, only warm hand-drawn graphics.

Scene: {your_description}.
"""