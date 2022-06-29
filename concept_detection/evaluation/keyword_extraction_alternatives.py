from rake_nltk import Rake
import RAKE
from keybert import KeyBERT

from utils.time.stopwatch import Stopwatch

text1 = """
    Théorie et critique du projet BA1Architecture projects ; envisioning environments , places of interaction , spaces , not as a product of immediate creation , but the result of a process where the tools of architectural production are critical . The 1st semester focusses on this field between actuality and the process of imagination . 
    Focussing on the relationship between the mind and the hand , students will enter an initial process , Arpenter , asking them to survey , map and cast . Beginning with one of four bridges located in Lausanne , students will devise both a method and an instrument to survey the site in relation to a given datum having defined which aspect of the site to survey and with what dimension it will be surveyed . The results are mapped onto a studio drawing in order to construct a mould and plaster cast , engaging the question of solid and void and therefore spatial thinking . The casts , which make reference to the given datum , are then set out as a mise-en-scène in the field , represented by a physical planar grid , and explored through a perspective drawing , from which notions of horizon , gravity and scale are debated . 
    The second phase , Habiter mon horizon , demands the students to investigate the relationship of an artefact such as a table , which both implicitly implies the body and a horizon , and a specific location on the site which they have surveyed . This relationship is then transcribed into a 1:1 site-specific construct , to be performed and debated in-situ . Expanding on this initial investigation , students will transcribe their project into a tectonic platform implying the relationship of body to space . Lectures will provide a scope of knowledge and perspective on architectural thinking as well as crafts . 
    Will be distributed at the beginning of the semester . 
"""

text2 = """
    Théorie et critique du projet BA2Architecture is what we would like the world to be . The 2nd semester provides a laboratory condition based on the field created in the 1st semester , where students expand on experiences , charting architectural process to develop their project , guided by the notions of tectonics , programme and space . 
    Continuing to work with the artefacts produced in the first semester , students will now engage in transcribing the relationships arising from the field that they have created into architectural projects . Beginning with the phase , Cadrages , each studetns defines a territory by intersecting the field with a paper periphery measuring 30x30x90cm , with the potential to overlap , enclosing a spatial idea that is to be extracted in the form of a wall , a slab and an opening , from whcih an architectural project will be elaborated . Through the tectonic , spatial and programmatic transformation of the paper periphery , students will question , by immersion , both its articulation as an enclosure and all that it encloses , investigating notions of interiority and exteriority , proximity and connectivity as well as negotiating any common teriitories . The dimension of this periphery also suggests verticality , as each spatial idea grows and extends structurally and infrastructurally to find ground . The semester outcome will be a matrix of spatially related towers - a small universe co-inhabited and created collectively by each studio . 
    Will be distributed at the beginning of the semester . 
"""

text3 = """
    Art du dessin IInitiation to the practice of architectural drawing through the various conventions of representation and graphic procedures , based on the use of paper and lead pencil . 
    The ABC of representation . 
    The line codes . 
    The various means of depiction (geometry , centring , full and empty space) . 
    Composition and layout . 
    Shadow and light . 
    Materials . Suggestion of textures . 
    Sketches and graphic notes . 
    geometry _ graphic procedures _ graphic expression _ graphic convention _ line
    Disclosed during the course .
"""


def rpyth(text):
    r = RAKE.Rake(RAKE.SmartStopList())
    keyword_list = r.run(text)
    # return [(s, k) for k, s in keyword_list]
    return [k for k, s in keyword_list]


def rnltk(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keyword_list = r.get_ranked_phrases_with_scores()
    # return keyword_list
    return [k for s, k in keyword_list]


def rbert(text):
    b = KeyBERT()
    keyword_list = b.extract_keywords(text, keyphrase_ngram_range=(1, 3), top_n=100, use_mmr=True, diversity=0.4)
    # return [(s, k) for k, s in keyword_list]
    return [k for k, s in keyword_list]


def venn(l1, l2, l3):
    s1 = set(l1)
    s2 = set(l2)
    s3 = set(l3)

    print(len(s1))
    print(len(s2))
    print(len(s3))

    print(len(s1 & s2 & s3))

    print(len(s1 & s2 - s3))
    print(len(s3 & s1 - s2))
    print(len(s2 & s3 - s1))

    print(len(s1 - s2 - s3))
    print(len(s2 - s3 - s1))
    print(len(s3 - s1 - s2))


text = text1

sw = Stopwatch()

rpyth_kw_list = rpyth(text)
print(f'pyth {sw.delta():.3f}s')

rnltk_kw_list = rnltk(text)
print(f'nltk {sw.delta():.3f}s')

rbert_kw_list = rbert(text)
print(f'bert {sw.delta():.3f}s')

venn(rpyth_kw_list, rnltk_kw_list, rbert_kw_list)

print(rpyth_kw_list)
print(rnltk_kw_list)
print(rbert_kw_list)

