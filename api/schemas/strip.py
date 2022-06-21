from pydantic import BaseModel, Field


class StripRequest(BaseModel):
    """
    Object containing the wikimarkdown code to be stripped.
    """
    markdown_code: str = Field(
        ...,
        title="Markdown code",
        description="Markdown code to be stripped",
        example="""'''Euclid''' ({{IPAc-en|ˈ|juː|k|l|ɪ|d}}; {{lang-grc-gre|[[Wikt:Εὐκλείδης|Εὐκλείδης]]}} {{transl|grc|Eukleides}}; {{fl.}} 300 BC), sometimes called '''Euclid of Alexandria'''<ref name=":0">{{Cite book|title=Math and Mathematicians: The History of Math Discoveries Around the World|last=Bruno|first=Leonard C.|date=2003|orig-year=1999|publisher=U X L|others=Baker, Lawrence W.|isbn=978-0-7876-3813-9|location=Detroit, Mich.|pages=[https://archive.org/details/mathmathematicia00brun/page/125 125]|oclc=41497065|url=https://archive.org/details/mathmathematicia00brun/page/125}}</ref> to distinguish him from [[Euclid of Megara]], was a [[Greek mathematics|Greek mathematician]], often referred to as the "founder of [[geometry]]"<ref name=":0" /> or the "father of geometry"."""
    )
