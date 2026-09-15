"""فکت‌های اوتیسم در ۷ دسته (A–G) به‌عنوان prototypeهای معنایی.

هر فکت دو نسخه‌ی انگلیسی و فارسی دارد. هنگام امبدینگ، دو نسخه به‌صورت دو prototype
مستقل در نظر گرفته می‌شوند تا فضای معنایی چندزبانه بهتر پوشش داده شود.
دسته‌ی G (دانش و آگاهی درباره‌ی اوتیسم) نیز در فاز لیبل‌گذاری داده استفاده می‌شود.

متن فارسی نسخه‌ی نهایی تأییدشده است؛ انگلیسی با همان معنا هم‌تراز شده.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: int
    category: str  # "A" .. "G"
    en: str
    fa: str


# --- تعریف دسته‌ها / Category metadata ---
CATEGORIES: dict[str, dict[str, str]] = {
    "A": {"key": "A", "title_en": "Social Communication & Interaction",
          "title_fa": "ارتباطات و تعامل اجتماعی"},
    "B": {"key": "B", "title_en": "Sensory Processing",
          "title_fa": "پردازش حسی"},
    "C": {"key": "C", "title_en": "Emotional Regulation & Stress",
          "title_fa": "تنظیم هیجان و استرس"},
    "D": {"key": "D", "title_en": "Routine & Predictability",
          "title_fa": "روتین و پیش‌بینی‌پذیری"},
    "E": {"key": "E", "title_en": "Special Interests & Strengths",
          "title_fa": "علایق ویژه و نقاط قوت"},
    "F": {"key": "F", "title_en": "Diagnosis & Support",
          "title_fa": "تشخیص و حمایت"},
    "G": {"key": "G", "title_en": "Autism Knowledge & Awareness",
          "title_fa": "دانش و آگاهی درباره اوتیسم"},
}

CATEGORY_KEYS: list[str] = list(CATEGORIES.keys())


FACTS: list[Fact] = [
    # ---------- A. Social Communication ----------
    Fact(1, "A",
         "People with autism spectrum disorder may experience challenges with social communication and interpreting nonverbal cues.",
         "افراد دارای اختلال طیف اوتیسم ممکن است در ارتباطات اجتماعی و درک نشانه‌های غیرکلامی چالش داشته باشند."),
    Fact(2, "A",
         "Eye contact can be difficult or exhausting for some people with autism spectrum disorder.",
         "تماس چشمی برای برخی افراد دارای اختلال طیف اوتیسم دشوار یا خسته‌کننده است."),
    Fact(3, "A",
         "Difficulty interpreting body language, tone of voice, sarcasm, and other social cues is common in autism.",
         "دشواری در درک زبان بدن، لحن صدا، کنایه و نشانه‌های اجتماعی در اوتیسم شایع است."),
    Fact(4, "A",
         "People with autism spectrum disorder often want social connection, but may communicate in ways that differ from others.",
         "افراد دارای اختلال طیف اوتیسم معمولاً به ارتباط علاقه دارند، اما ممکن است شیوه برقراری ارتباط آن‌ها متفاوت باشد."),
    Fact(5, "A",
         "Spoken-language abilities among people with autism spectrum disorder vary widely, from people who do not use spoken language to those who communicate using sentences.",
         "مهارت‌های کلامی در افراد دارای اختلال طیف اوتیسم بسیار متنوع است؛ از افراد غیرکلامی تا افراد دارای توانایی بیان جمله."),
    Fact(6, "A",
         "People with autism spectrum disorder experience emotions and empathy, but may express them differently from others.",
         "افراد دارای اختلال طیف اوتیسم احساسات و همدلی دارند، اما ممکن است آن را متفاوت از دیگران ابراز کنند."),

    # ---------- B. Sensory Processing ----------
    Fact(7, "B",
         "Many people with autism spectrum disorder have differences in sensory processing.",
         "بسیاری از افراد دارای اختلال طیف اوتیسم تفاوت‌هایی در پردازش حسی دارند."),
    Fact(8, "B",
         "Hyper- or hyporeactivity to sensory input, including sound, light, touch, smell, or other stimuli, is common in autism.",
         "حساسیت بیش از حد یا کمتر از حد معمول به صدا، نور، لمس، بو یا سایر محرک‌ها شایع است."),
    Fact(9, "B",
         "Sensory overload can lead to anxiety, exhaustion, or severe distress.",
         "اضافه‌بار حسی (Sensory Overload) می‌تواند موجب اضطراب، خستگی یا ناراحتی شدید شود."),

    # ---------- C. Emotional Regulation & Stress ----------
    Fact(10, "C",
         "Repetitive behaviors, often referred to as stimming, can help with emotional regulation or stress reduction.",
         "رفتارهای تکراری (Stimming) اغلب به تنظیم هیجان یا کاهش استرس کمک می‌کنند."),
    Fact(11, "C",
         "Meltdowns, shutdowns, or intense distress can occur when emotions become overwhelming.",
         "وقتی هیجان‌ها بیش از حد شدید می‌شوند ممکن است فروپاشی هیجانی، خاموشی یا ناراحتی شدید رخ دهد."),
    Fact(12, "C",
         "People with autism spectrum disorder may need additional time and support to recover after emotional overload.",
         "افراد دارای اختلال طیف اوتیسم ممکن است پس از اضافه‌بار هیجانی به زمان و حمایت بیشتری برای بازگشت به تعادل نیاز داشته باشند."),
    Fact(13, "C",
         "Anxiety, frustration, or shutdown responses can be reduced through access to calm environments and emotion-regulation strategies.",
         "اضطراب، ناکامی یا خاموشی هیجانی را می‌توان با فضای آرام و راهبردهای تنظیم هیجان کاهش داد."),

    # ---------- D. Routine & Predictability ----------
    Fact(14, "D",
         "Many people with autism spectrum disorder need consistency and predictability to feel safe.",
         "بسیاری از افراد دارای اختلال طیف اوتیسم برای احساس امنیت به ثبات و پیش‌بینی‌پذیری نیاز دارند."),
    Fact(15, "D",
         "Sudden changes in plans or the environment can be very difficult for some people with autism spectrum disorder.",
         "تغییر ناگهانی برنامه‌ها یا محیط می‌تواند برای برخی افراد دارای اختلال طیف اوتیسم بسیار دشوار باشد."),
    Fact(16, "D",
         "A preference for clear, predictable routines is common among people on the autism spectrum.",
         "ترجیح روتین‌های مشخص یکی از ویژگی‌های رایج در طیف اوتیسم است."),

    # ---------- E. Special Interests & Strengths ----------
    Fact(17, "E",
         "Many people with autism spectrum disorder have highly focused or unusually intense interests.",
         "بسیاری از افراد دارای اختلال طیف اوتیسم علایق بیش از حد غیرمعمول دارند."),
    Fact(18, "E",
         "Some people with autism spectrum disorder show remarkable skill or expertise in specific areas.",
         "برخی افراد دارای اختلال طیف اوتیسم در زمینه‌های خاص مهارت یا تخصص چشمگیری دارند."),
    Fact(19, "E",
         "Sustained focus and strong attention to detail are seen in some people with autism spectrum disorder.",
         "تمرکز طولانی‌مدت و توجه زیاد به جزئیات در برخی افراد دارای اختلال طیف اوتیسم دیده می‌شود."),
    Fact(20, "E",
         "Autism does not mean low intelligence.",
         "اوتیسم به معنای کم‌هوشی نیست."),
    Fact(21, "E",
         "People with autism spectrum disorder can have different levels of intellectual ability.",
         "افراد دارای اختلال طیف اوتیسم می‌توانند بهره‌های هوشی متفاوتی داشته باشند."),
    Fact(22, "E",
         "Some people with autism spectrum disorder who have lower support needs may live independently, pursue education, work, and have relationships.",
         "برخی افراد دارای اختلال طیف اوتیسم سطح بالا ممکن است زندگی مستقل، تحصیل، اشتغال و روابط عادی داشته باشند."),
    Fact(23, "E",
         "Not all people with autism spectrum disorder have savant syndrome or exceptional talents.",
         "همه افراد دارای اختلال طیف اوتیسم دارای استعداد خارق‌العاده یا سندرم ساوان نیستند."),

    # ---------- F. Diagnosis & Support ----------
    Fact(24, "F",
         "Signs of autism typically emerge in early childhood.",
         "علائم اوتیسم معمولاً در اوایل کودکی ظاهر می‌شوند."),
    Fact(25, "F",
         "Diagnosis of autism is usually based on behavioral assessment and developmental history.",
         "تشخیص اوتیسم معمولاً بر اساس ارزیابی رفتاری و تاریخچه رشد انجام می‌شود."),
    Fact(26, "F",
         "There is no blood test or single definitive medical test for diagnosing autism.",
         "هیچ آزمایش خون یا تست پزشکی قطعی برای تشخیص اوتیسم وجود ندارد."),
    Fact(27, "F",
         "Early diagnosis can help improve long-term outcomes.",
         "تشخیص زودهنگام می‌تواند به بهبود نتایج زندگی کمک کند."),
    Fact(28, "F",
         "Supportive and educational interventions, speech-language therapy, and occupational therapy can be helpful.",
         "مداخلات حمایتی، آموزشی، گفتاردرمانی و کاردرمانی می‌توانند مفید باشند."),
    Fact(29, "F",
         "The goal of support is to improve quality of life and independence, not to change a person's identity.",
         "هدف حمایت‌ها افزایش کیفیت زندگی و استقلال فرد است، نه تغییر هویت او."),

    # ---------- G. Autism Knowledge & Awareness ----------
    Fact(30, "G",
         "Autism is a neurodevelopmental condition, not a mental illness.",
         "اوتیسم یک وضعیت عصبی-رشدی است، نه یک بیماری روانی."),
    Fact(31, "G",
         "Autism is part of human neurodiversity.",
         "اوتیسم بخشی از تنوع عصبی (Neurodiversity) انسان محسوب می‌شود."),
    Fact(32, "G",
         "Autism is a spectrum, and people with autism spectrum disorder can have widely differing traits and needs.",
         "اوتیسم یک طیف است و افراد دارای اختلال طیف اوتیسم می‌توانند ویژگی‌ها و نیازهای بسیار متفاوتی داشته باشند."),
    Fact(33, "G",
         "Autism can co-occur with other conditions or disabilities, including low vision or blindness, hearing loss or deafness, cerebral palsy, and intellectual disability.",
         "اوتیسم می‌تواند همراه با سایر شرایط یا معلولیت‌ها، مانند کم‌بینایی یا نابینایی، کم‌شنوایی یا ناشنوایی، فلج مغزی یا ناتوانی ذهنی، وجود داشته باشد."),
    Fact(34, "G",
         "Autism is a lifelong condition.",
         "اوتیسم یک وضعیت مادام‌العمر است."),
    Fact(35, "G",
         "Autism does not have a specific physical appearance.",
         "اوتیسم ظاهر فیزیکی مشخصی ندارد."),
    Fact(36, "G",
         "The exact causes of autism are not yet fully understood.",
         "علت دقیق اوتیسم هنوز کاملاً مشخص نیست."),
    Fact(37, "G",
         "Genetic factors play an important role in the development of autism.",
         "عوامل ژنتیکی نقش مهمی در بروز اوتیسم دارند."),
    Fact(38, "G",
         "Some environmental factors may also contribute alongside genetic factors.",
         "برخی عوامل محیطی ممکن است در کنار ژنتیک نقش داشته باشند."),
    Fact(39, "G",
         "Autism is not caused by parenting style or parental behavior.",
         "اوتیسم نتیجه سبک فرزندپروری یا رفتار والدین نیست."),
    Fact(40, "G",
         "Autism is diagnosed more frequently in boys than in girls.",
         "اوتیسم در پسران بیشتر از دختران تشخیص داده می‌شود."),
    Fact(41, "G",
         "Some girls with autism spectrum disorder may be underdiagnosed, in part because autistic traits can be masked or camouflaged.",
         "برخی دختران دارای اختلال طیف اوتیسم ممکن است به دلیل ماسکینگ کمتر تشخیص داده شوند."),
    Fact(42, "G",
         "Autism occurs across all genders, races, countries, social and cultural groups, and socioeconomic levels.",
         "اوتیسم در همه جنسیت‌ها، نژادها، کشورها، گروه‌های اجتماعی و فرهنگی و سطوح مختلف اجتماعی-اقتصادی دیده می‌شود."),
    Fact(43, "G",
         "Vaccines do not cause autism.",
         "واکسن‌ها باعث اوتیسم نمی‌شوند."),
    Fact(44, "G",
         "Respect for neurodiversity and reduction of social stigma are important.",
         "احترام به تفاوت‌های عصبی و کاهش انگ اجتماعی اهمیت دارد."),
]


def get_prototypes(lang: str = "both") -> list[dict]:
    """برگرداندن prototypeها به‌صورت لیست دیکشنری برای امبدینگ.

    Args:
        lang: "en" | "fa" | "both"
            - both: هر فکت دو prototype (EN و FA) می‌سازد (رفتار task اول).
            - en / fa: فقط همان زبان برای مقایسه‌ی کنترل‌شده‌ی چندمدلی.
    """
    lang = (lang or "both").lower().strip()
    if lang not in {"en", "fa", "both"}:
        raise ValueError(f"Unsupported fact lang '{lang}'. Use en|fa|both.")

    protos: list[dict] = []
    for f in FACTS:
        if lang in ("en", "both"):
            protos.append(
                {"fact_id": f.id, "category": f.category, "lang": "en", "text": f.en}
            )
        if lang in ("fa", "both"):
            protos.append(
                {"fact_id": f.id, "category": f.category, "lang": "fa", "text": f.fa}
            )
    return protos


def category_titles_bilingual() -> dict[str, str]:
    return {k: f"{v['key']}. {v['title_en']} — {v['title_fa']}"
            for k, v in CATEGORIES.items()}


if __name__ == "__main__":
    protos = get_prototypes()
    print(f"Total facts: {len(FACTS)} | Total prototypes (en+fa): {len(protos)}")
    for k in CATEGORY_KEYS:
        n = sum(1 for f in FACTS if f.category == k)
        print(f"  {k}: {n} facts -> {n*2} prototypes | {CATEGORIES[k]['title_en']}")
