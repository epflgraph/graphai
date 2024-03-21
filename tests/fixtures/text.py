import pandas as pd
import pytest


@pytest.fixture
def sultans():
    return """
        <p>
        Then a crowd a young boys they're a foolin' around in the corner
        Drunk and dressed in their best brown baggies and their platform soles
        They don't give a damn about any trumpet playin' band
        It ain't what they call 'rock and roll'
        </p>
    """


@pytest.fixture
def wave_fields():
    return """Consider a nonparametric representation of acoustic wave fields that consists of observing the sound pressure along a straight line or a smooth contour L defined in space. The observed data contains implicit information of the surrounding acoustic scene, both in terms of spatial arrangement of the sources and their respective temporal evolution. We show that such data can be effectively analyzed and processed in what we call the space-time-frequency representation space, consisting of a Gabor representation across the spatio-temporal manifold defined by the spatial axis L and the temporal axis t. In the presence of a source, the spectral patterns generated at L have a characteristic triangular shape that changes according to certain parameters, such as the source distance and direction, the number of sources, the concavity of L, and the analysis window size. Yet, in general, the wave fronts can be expressed as a function of elementary directional components-most notably, plane waves and far-field components. Furthermore, we address the problem of processing the wave field in discrete space and time, i.e., sampled along L and t, where a Gabor representation implies that the wave fronts are processed in a block-wise fashion. The key challenge is how to chose and customize a spatio-temporal filter bank such that it exploits the physical properties of the wave field while satisfying strict requirements such as perfect reconstruction, critical sampling, and computational efficiency. We discuss the architecture of such filter banks, and demonstrate their applicability in the context of real applications, such as spatial filtering, deconvolution, and wave field coding."""


@pytest.fixture
def schreier():
    return """We are interested in various aspects of spectral rigidity of Cayley and Schreier graphs of finitely generated groups. For each pair of integers, we consider an uncountable family of groups of automorphisms of the rooted d-regular tree, which provide examples of the following interesting phenomena. We get an uncountable family of non-quasi-isometric Cayley graphs with the same Laplacian spectrum, a union of two intervals, which we compute explicitly. Some of the groups provide examples where the spectrum of the Cayley graph is connected for one generating set and has a gap for another. We exhibit infinite Schreier graphs of these groups with the spectrum a Cantor set of Lebesgue measure zero union a countable set of isolated points accumulating on it. The Kesten spectral measures of the Laplacian on these Schreier graphs are discrete and concentrated on the isolated points. We construct, moreover, a complete system of eigenfunctions that are strongly localized."""


@pytest.fixture
def euclid():
    return ["straight line", "point to point", "describe a circle", "all right angles equal", "two straight lines", "interior angles", "less than two right angles"]


@pytest.fixture
def wave_fields_wikisearch_df():
    return pd.DataFrame([
        ['acoustic wave fields', 1901541, """Ion acoustic wave""", 1, 1],
        ['acoustic wave fields', 459844, """Surface acoustic wave""", 2, 0.888889],
        ['acoustic wave fields', 144940, """Longitudinal wave""", 3, 0.777778],
        ['acoustic wave fields', 6101054, """Acoustic levitation""", 4, 0.666667],
        ['acoustic wave fields', 33516, """Wave""", 5, 0.555556],
        ['acoustic wave fields', 24476128, """Acoustic metamaterial""", 6, 0.444444],
        ['acoustic wave fields', 299813, """Acoustical engineering""", 7, 0.333333],
        ['acoustic wave fields', 5786179, """Acoustic wave""", 8, 0.222222],
        ['acoustic wave fields', 35537980, """Acoustic attenuation""", 9, 0.111111],
        ['acoustic wave fields', 2630105, """Acoustic wave equation""", 10, 0]
    ], columns=['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'])


@pytest.fixture
def wave_fields_wikified_json():
    return [
        {
            'concept_id': '459844',
            'concept_name': """Surface acoustic wave""",
            'search_score': 0.888889,
            'levenshtein_score': 0.750277,
            'graph_score': 0.086486,
            'ontology_local_score': 0.2375,
            'ontology_global_score': 0.2375,
            'keywords_score': 1.0,
            'mixed_score': 0.658343,
        },
        {
            'concept_id': '144940',
            'concept_name': """Longitudinal wave""",
            'search_score': 0.777778,
            'levenshtein_score': 0.367288,
            'graph_score': 0.913514,
            'ontology_local_score': 0.7125,
            'ontology_global_score': 0.7125,
            'keywords_score': 1.0,
            'mixed_score': 0.780125,
        },
        {
            'concept_id': '33516',
            'concept_name': """Wave""",
            'search_score': 0.555556,
            'levenshtein_score': 0.200000,
            'graph_score': 1.000000,
            'ontology_local_score': 0.7125,
            'ontology_global_score': 0.7125,
            'keywords_score': 1.0,
            'mixed_score': 0.719236,
        },
    ]
