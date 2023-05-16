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
def wave_fields_wikisearch_df():
    return pd.DataFrame([
        ['acoustic wave fields', 1901541, """Ion_acoustic_wave""", 1, 1, 0.848181],
        ['acoustic wave fields', 459844, """Surface_acoustic_wave""", 2, 0.888889, 0.750277],
        ['acoustic wave fields', 144940, """Longitudinal_wave""", 3, 0.777778, 0.367288],
        ['acoustic wave fields', 6101054, """Acoustic_levitation""", 4, 0.666667, 0.626132],
        ['acoustic wave fields', 33516, """Wave""", 5, 0.555556, 0.200000],
        ['acoustic wave fields', 24476128, """Acoustic_metamaterial""", 6, 0.444444, 0.750277],
        ['acoustic wave fields', 299813, """Acoustical_engineering""", 7, 0.333333, 0.640000],
        ['acoustic wave fields', 5786179, """Acoustic_wave""", 8, 0.222222, 0.932414],
        ['acoustic wave fields', 35537980, """Acoustic_attenuation""", 9, 0.111111, 0.692308],
        ['acoustic wave fields', 2630105, """Acoustic_wave_equation""", 10, 0, 0.862069]
    ], columns=['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'LevenshteinScore'])


@pytest.fixture
def wave_fields_scores_df():
    return pd.DataFrame([
        ['acoustic wave fields', 459844, """Surface_acoustic_wave""", 2, 0.888889, 0.750277, 527, 111, 0.2375, 0.2375, 0.086486, 1.0],
        ['acoustic wave fields', 144940, """Longitudinal_wave""", 3, 0.777778, 0.367288, 974, 194, 0.7125, 0.7125, 0.913514, 1.0],
        ['acoustic wave fields', 33516, """Wave""", 4, 0.555556, 0.200000, 974, 194, 0.7125, 0.7125, 1.000000, 1.0]
    ], columns=['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'LevenshteinScore', 'CategoryID', 'Category2ID', 'OntologyLocalScore', 'OntologyGlobalScore', 'GraphScore', 'KeywordsScore'])
