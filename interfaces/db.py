import mysql.connector
import configparser

from definitions import CONFIG_DIR

from models.wikify_result import WikifyResult


class DB:
    """
    Base class to communicate with the EPFLGraph database.
    """

    def __init__(self):
        # Read db config from file and open connection
        db_config = configparser.ConfigParser()
        db_config.read(f'{CONFIG_DIR}/db.ini')

        self.host = db_config['DB'].get('host')
        self.port = db_config['DB'].getint('port')
        self.user = db_config['DB'].get('user')
        self.password = db_config['DB'].get('password')

        self.connect()

        # Fetch initial data
        self.channel_anchor_page_ids = self.query_channel_anchor_page_ids()
        self.course_channel_ids = self.query_course_channel_ids()

    def connect(self):
        self.cnx = mysql.connector.connect(host=self.host, port=self.port, user=self.user, password=self.password)
        self.cursor = self.cnx.cursor()

    def query(self, query):
        """
        Execute custom query.
        """

        self.connect()

        self.cursor.execute(query)

        return list(self.cursor)

    def query_channel_anchor_page_ids(self):
        """
        Retrieve a mapping from each SWITCH channel id to its list of anchor page ids.

        Returns:
            dict[str, list[int]]: A dictionary with SWITCH channel ids as keys and list of anchor page ids as values.
        """

        self.connect()

        query = f"""
            SELECT SwitchChannelID, IF(AnchorPageIDs1 IS NOT NULL, AnchorPageIDs1, AnchorPageIDs2) AS AnchorPageIDs
            FROM (
                SELECT DISTINCT m.SwitchChannelID,
                    m.CourseID, SUBSTRING_INDEX(m.CourseID, '-', 1) AS CoursePrefix, 
                    GROUP_CONCAT(PageID SEPARATOR ',') AS AnchorPageIDs1, 
                    p.AnchorPageIDs AS AnchorPageIDs2
                FROM ca_switchtube.Channel_to_Course_Mapping m
                LEFT JOIN (
                    SELECT m.CourseCode, m.Keywords, m.PageID, m.PageTitle
                    FROM man_isacademia.Course2Wiki_Current_Mapping m
                    INNER JOIN (
                        SELECT CourseCode, Keywords, MAX(Revision) AS LatestRevision
                        FROM man_isacademia.Course2Wiki_Current_Mapping
                        GROUP BY CourseCode, Keywords
                    ) r
                    USING (CourseCode, Keywords)
                    WHERE m.Revision=r.LatestRevision
                ) t
                ON CourseID=CourseCode
                LEFT JOIN man_switchtube.CoursePrefix_Top16_Pages p 
                ON SUBSTRING_INDEX(m.CourseID, '-', 1)=p.CoursePrefix
                GROUP BY SwitchChannelID
            ) t
            WHERE NOT (AnchorPageIDs1 IS NULL AND AnchorPageIDs2 IS NULL);
        """
        self.cursor.execute(query)

        channel_anchor_page_ids = {}
        for channel_id, anchor_page_ids in self.cursor:
            anchor_page_ids = [int(s) for s in anchor_page_ids.split(',')]
            channel_anchor_page_ids[channel_id] = anchor_page_ids

        return channel_anchor_page_ids

    def query_slide_texts(self, limit=10, offset=0, pseudorandom=False):
        """
        Retrieve a mapping from each slide id to its OCR-extracted text.

        Returns:
            dict[str, str]: A dictionary with slide ids as keys and slide texts as values.
        """

        self.connect()

        query = f"""
            SELECT DISTINCT st.SlideID, st.SlideText
            FROM gen_switchtube.Slide_Text st
            INNER JOIN gen_switchtube.Slide_Keywords sk
            ON st.SlideID = sk.SlideID
        """

        if pseudorandom:
            query += f"""
                WHERE st.SlideID REGEXP "[0-9|a-z|A-Z]*_[0-9|a-z|A-Z]*_[0-9|a-z|A-Z]*[9][9][0-9|a-z|A-Z]*"
            """

        query += f"""
            LIMIT {limit} OFFSET {offset};
        """
        self.cursor.execute(query)

        slide_texts = {}
        for slide_id, slide_text in self.cursor:
            slide_texts[slide_id] = slide_text

        return slide_texts

    def query_wikified_slides(self, slide_ids):
        """
        Retrieve the results of wikifying the given slides, as stored in the database.

        Args:
            slide_ids (list[str]): List of slide ids.

        Returns:
            list[dict[str]]: List of wikify results for keywords from the texts of the given slides.
            Each element has the following keys:

            * 'keywords' (str): Set of keywords.
            * 'page_id' (int): Id of the wikipage.
            * 'page_title' (str): Title of the wikipage.
            * 'searchrank' (int): Position of the wikipage in the wikisearch.
            * 'median_graph_score' (float): Median of the graph_score over all anchor pages.
            * 'searchrank_graph_ratio' (float): Ratio graph_score/search_score.
        """

        self.connect()

        query = f"""
            SELECT DISTINCT sk.SlideID, sk.Keywords, sk.PageID, sk.PageTitle, sk.Rank, sk.Score 
            FROM gen_switchtube.Slide_Text st
            INNER JOIN gen_switchtube.Slide_Keywords sk
            ON st.SlideID = sk.SlideID
            WHERE sk.SlideID IN ({', '.join(['%s']*len(slide_ids))});
        """
        self.cursor.execute(query, slide_ids)

        wikified_slides = {}
        for slide_id, keywords, page_id, page_title, searchrank, score in self.cursor:
            if slide_id not in wikified_slides:
                wikified_slides[slide_id] = []

            wikified_slides[slide_id].append({
                'keywords': keywords,
                'page_id': page_id,
                'page_title': page_title,
                'searchrank': searchrank,
                'median_graph_score': score * searchrank,
                'searchrank_graph_ratio': score
            })

        return wikified_slides

    def slide_anchor_page_ids(self, slide_id):
        """
        Retrieve list of anchor page ids for the given slide.

        Args:
            slide_id (int): Id of a slide.

        Returns:
            list[int]: List of ids of the anchor pages associated with the given slide.
        """

        self.connect()

        channel_id = slide_id.split('_')[0]
        return self.channel_anchor_page_ids[channel_id]

    def query_course_descriptions(self):
        """
        Retrieve a mapping from each course id to its description text.

        Returns:
            dict[str, str]: A dictionary with course ids as keys and course descriptions as values.
        """

        self.connect()

        query = f"""
            SELECT CourseCode, CONCAT_WS(
                '\n', SubjectName, Abstract, Contents, Keywords, Concepts, Bibliography
            ) AS Description
            FROM (
                SELECT
                    CourseCode, SubjectName,
                    MIN(CASE WHEN FieldType = 'Résumé' THEN FieldValue END) AS Abstract,
                    MIN(CASE WHEN FieldType = 'Contenu' THEN FieldValue END) AS Contents,
                    MIN(CASE WHEN FieldType = 'Mots-clés' THEN FieldValue END) AS Keywords,
                    MIN(CASE WHEN FieldType = 'Concepts importants à maîtriser' THEN FieldValue END) AS Concepts,
                    MIN(CASE WHEN FieldType = 'Bibliographie' THEN FieldValue END) AS Bibliography
                FROM man_isacademia.Course2Wiki_Simplified_Descriptions
                GROUP BY CourseCode
            ) t
            INNER JOIN (
                SELECT DISTINCT CourseCode
                FROM man_isacademia.Course2Wiki_Current_Mapping
            ) s
            USING (CourseCode)
        """
        self.cursor.execute(query)

        course_descriptions = {}
        for course_id, course_description in self.cursor:
            course_descriptions[course_id] = course_description

        return course_descriptions

    def query_wikified_course_descriptions(self):
        """
        Retrieve the results of wikifying the given course descriptions, as stored in the database.

        Returns:
            list[dict[str]]: List of wikify results for keywords from the given course descriptions.
            Each element has the following keys:

            * 'keywords' (str): Set of keywords.
            * 'page_id' (int): Id of the wikipage.
            * 'page_title' (str): Title of the wikipage.
            * 'searchrank' (int): Position of the wikipage in the wikisearch.
            * 'median_graph_score' (float): Median of the graph_score over all anchor pages.
            * 'searchrank_graph_ratio' (float): Ratio graph_score/search_score.
        """

        self.connect()

        query = f"""
            SELECT m.CourseCode, m.Keywords, m.PageID, m.PageTitle
            FROM man_isacademia.Course2Wiki_Current_Mapping m
            INNER JOIN (
                SELECT CourseCode, Keywords, MAX(Revision) AS LatestRevision
                FROM man_isacademia.Course2Wiki_Current_Mapping
                GROUP BY CourseCode, Keywords
            ) r
            USING (CourseCode, Keywords)
            INNER JOIN (
                SELECT DISTINCT CourseCode
                FROM man_isacademia.Course2Wiki_Simplified_Descriptions
            ) s
            USING (CourseCode)
            WHERE m.Revision=r.LatestRevision
        """
        self.cursor.execute(query)

        wikified_course_descriptions = {}
        for course_id, keywords, page_id, page_title in self.cursor:
            if course_id not in wikified_course_descriptions:
                wikified_course_descriptions[course_id] = []

            wikified_course_descriptions[course_id].append(WikifyResult(
                keywords=keywords,
                page_id=page_id,
                page_title=page_title,
                median_graph_score=1
            ))

        return wikified_course_descriptions

    def query_course_channel_ids(self):
        """
        Retrieve a mapping from each course id to its list of channel ids.

        Returns:
            dict[str, str]: A dictionary with course ids as keys and lists of their associated channel ids as values.
        """

        self.connect()

        query = f"""
            SELECT CourseID, GROUP_CONCAT(SwitchChannelID SEPARATOR ',') AS SwitchChannelIDs
            FROM ca_switchtube.Channel_to_Course_Mapping
            WHERE CourseID IS NOT NULL
            AND CourseID != 'n/a'
            GROUP BY CourseID
        """
        self.cursor.execute(query)

        course_channel_ids = {}
        for course_ids, channel_ids in self.cursor:
            channel_ids = [s for s in channel_ids.split(',')]
            course_channel_ids[course_ids] = channel_ids

        return course_channel_ids

    def course_anchor_page_ids(self, course_id):
        """
        Retrieve list of anchor page ids for the given course.

        Args:
            course_id (str): Id of a course.

        Returns:
            list[int]: List of ids of the anchor pages associated with the given course.
        """

        self.connect()

        channel_ids = self.course_channel_ids.get(course_id, None)
        if channel_ids:
            return list({anchor_page_id for channel_id in channel_ids for anchor_page_id in self.channel_anchor_page_ids[channel_id]})
        else:
            return []

    def get_wikipage_ids(self, filter_orphan=False):
        """
        Retrieve a full list of all wikipage ids present in the database.

        Args:
            filter_orphan (bool): Whether to filter out wikipages with no links from or to other wikipages.

        Returns:
            list[int]: List of wikipage ids.
        """

        self.connect()

        if filter_orphan:
            query = f"""
                SELECT DISTINCT SourcePageID
                FROM piper_wikipedia.Page_Links_Random_Walk
            """
            self.cursor.execute(query)

            source_page_ids = []
            for source_page_id, in self.cursor:
                source_page_ids.append(source_page_id)

            query = f"""
                SELECT DISTINCT TargetPageID
                FROM piper_wikipedia.Page_Links_Random_Walk
            """
            self.cursor.execute(query)

            target_page_ids = []
            for target_page_id, in self.cursor:
                target_page_ids.append(target_page_id)

            return list(set(source_page_ids) | set(target_page_ids))

        else:
            query = f"""
                SELECT PageID
                FROM piper_wikipedia.PageTitle_to_PageID_Mapping
            """
            self.cursor.execute(query)

            page_ids = []
            for page_id, in self.cursor:
                page_ids.append(page_id)

            return page_ids

    def get_wikipages(self, ids=None, id_min_max=None, limit=None):
        """
        Retrieve wikipages and related information from the database.

        Args:
            ids (list[int]): If set, restrict to wikipages with id in this list.
            id_min_max (tuple[int, int]): If set, restrict to wikipages with id in the given range.
            limit: If set, limit the number of wikipages to this number.

        Returns:
            dict[int, dict[str]]: Dictionary with wikipage ids as keys and whose values contain the following keys:

            * 'title' (str): Title of the wikipage.
            * 'content' (str): Content of the wikipage.
            * 'redirect' (list[str]): List of titles of wikipages redirecting to this one.
            * 'popularity' (float): Popularity score of the wikipage.
        """

        self.connect()

        query = """
            SELECT titles.PageID, titles.PageTitle, contents.PageContent, contents.Redirects, contents.Popularity
            FROM piper_wikipedia.PageTitle_to_PageID_Mapping AS titles
            INNER JOIN piper_wikipedia.Page_Content AS contents
            ON titles.PageID = contents.PageID
        """

        conditions = []

        if ids is not None:
            ids = [str(_id) for _id in ids]
            conditions.append(f"""titles.PageID IN ({','.join(ids)})""")

        if id_min_max is not None:
            conditions.append(f"""titles.PageID >= {id_min_max[0]}""")
            conditions.append(f"""titles.PageID < {id_min_max[1]}""")

        if conditions:
            query += f"""
                WHERE {' AND '.join(conditions)}
            """

        if limit is not None:
            query += f"""
                LIMIT {limit}
            """

        self.cursor.execute(query)

        return {
            _id: {
                'title': title,
                'content': content,
                'redirect': redirect if redirect else [],
                'popularity': popularity if popularity else 1
            }
            for _id, title, content, redirect, popularity in self.cursor
        }

    def get_wikipage_categories(self, ids=None, id_min_max=None, limit=None):
        """
        Retrieve wikipage categories from the database.

        Args:
            ids (list[int]): If set, restrict to wikipages with id in this list.
            id_min_max (tuple[int, int]): If set, restrict to wikipages with id in the given range.
            limit: If set, limit the number of wikipages to this number.

        Returns:
            dict[int, str]: Dictionary with wikipage ids as keys and a concatenation of the category titles as values.
        """

        self.connect()

        query = """            
            SELECT titles.PageID, GROUP_CONCAT(categories.CategoryTitle)
            FROM piper_wikipedia.PageTitle_to_PageID_Mapping AS titles
            INNER JOIN piper_wikipedia.PageID_to_CategoriesID_Mapping AS pages_categories
            ON titles.PageId = pages_categories.PageID
            INNER JOIN piper_wikipedia.Categories AS categories
            ON pages_categories.CategoryID = categories.CategoryID
        """

        conditions = []

        if ids is not None:
            ids = [str(_id) for _id in ids]
            conditions.append(f"""pages_categories.PageID IN ({','.join(ids)})""")

        if id_min_max is not None:
            conditions.append(f"""pages_categories.PageID >= {id_min_max[0]}""")
            conditions.append(f"""pages_categories.PageID < {id_min_max[1]}""")

        if conditions:
            query += f"""
                        WHERE {' AND '.join(conditions)}
                    """

        query += """
            GROUP BY titles.PageID
        """

        if limit is not None:
            query += f"""
                LIMIT {limit}
            """

        self.cursor.execute(query)

        return {
            page_id: categories.split(',')
            for page_id, categories in self.cursor
        }

    def get_table(self, table, fields=None, limit=None):
        """
        TODO DOC
        Args:
            table:

        Returns:
        """

        self.connect()

        if fields is None:
            query = f"""
                SELECT * FROM {table}
            """
        else:
            query = f"""
                SELECT {', '.join([field for field in fields])} FROM {table}
            """

        if limit is not None:
            query += f"""LIMIT {limit}"""

        return self.query(query)

    def get_investees_funding_rounds(self, org_ids=None, fr_ids=None):

        self.connect()

        query = f"""
            SELECT OrganisationID, FundingRoundID 
            FROM graph.Edges_N_Organisation_N_FundingRound
        """

        conditions = ['Action = "Raised from"']
        ids = []

        if org_ids is not None:
            conditions.append(f"""OrganisationID IN ({', '.join(['%s'] * len(org_ids))})""")
            ids.extend(org_ids)

        if fr_ids is not None:
            conditions.append(f"""FundingRoundID IN ({', '.join(['%s'] * len(fr_ids))})""")
            ids.extend(fr_ids)

        query += f"""
            WHERE {' AND '.join(conditions)}
        """

        if ids:
            self.cursor.execute(query, ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_org_investors_funding_rounds(self, org_ids=None, fr_ids=None):

        self.connect()

        query = f"""
            SELECT OrganisationID, FundingRoundID 
            FROM graph.Edges_N_Organisation_N_FundingRound
        """

        conditions = ['Action = "Invested in"']
        ids = []

        if org_ids is not None:
            conditions.append(f"""OrganisationID IN ({', '.join(['%s'] * len(org_ids))})""")
            ids.extend(org_ids)

        if fr_ids is not None:
            conditions.append(f"""FundingRoundID IN ({', '.join(['%s'] * len(fr_ids))})""")
            ids.extend(fr_ids)

        query += f"""
            WHERE {' AND '.join(conditions)}
        """

        if ids:
            self.cursor.execute(query, ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_person_investors_funding_rounds(self, person_ids=None, fr_ids=None):

        self.connect()

        query = f"""
            SELECT PersonID, FundingRoundID 
            FROM graph.Edges_N_Person_N_FundingRound
        """

        conditions = ['Action = "Invested in"']
        ids = []

        if person_ids is not None:
            conditions.append(f"""PersonID IN ({', '.join(['%s'] * len(person_ids))})""")
            ids.extend(person_ids)

        if fr_ids is not None:
            conditions.append(f"""FundingRoundID IN ({', '.join(['%s'] * len(fr_ids))})""")
            ids.extend(fr_ids)

        query += f"""
            WHERE {' AND '.join(conditions)}
        """

        if ids:
            self.cursor.execute(query, ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_concepts_organisations(self, concept_ids=None, org_ids=None):

        self.connect()

        query = f"""
            SELECT PageID, OrganisationID 
            FROM graph.Edges_N_Organisation_N_Concept
        """

        conditions = []
        ids = []

        if concept_ids is not None:
            conditions.append(f"""PageID IN ({', '.join(['%s'] * len(concept_ids))})""")
            ids.extend(concept_ids)

        if org_ids is not None:
            conditions.append(f"""OrganisationID IN ({', '.join(['%s'] * len(org_ids))})""")
            ids.extend(org_ids)

        if conditions:
            query += f"""
                WHERE {' AND '.join(conditions)}
            """

        if ids:
            self.cursor.execute(query, ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_funding_rounds(self, min_date=None, max_date=None, fr_ids=None, fields=None):

        self.connect()

        if fields is None:
            query = f"""
                SELECT FundingRoundID,
                       CONCAT(YEAR(FundingRoundDate), "-Q", QUARTER(FundingRoundDate)) AS FundingRoundTimePeriod,
                       FundingAmount_USD
                FROM graph.Nodes_N_FundingRound
            """
        else:
            query = f"""
                SELECT {', '.join(fields)}
                FROM graph.Nodes_N_FundingRound
            """

        conditions = []

        if min_date is not None:
            conditions.append(f"""FundingRoundDate >= "{min_date}" """)

        if max_date is not None:
            conditions.append(f"""FundingRoundDate < "{max_date}" """)

        if fr_ids is not None:
            conditions.append(f"""FundingRoundID IN ({', '.join(['%s'] * len(fr_ids))})""")

        if conditions:
            query += f"""
                WHERE {' AND '.join(conditions)}
            """

        if fr_ids is not None:
            self.cursor.execute(query, fr_ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_organisations(self, org_ids=None):

        self.connect()

        query = f"""
            SELECT OrganisationID,
                   OrganisationName
            FROM graph.Nodes_N_Organisation
        """

        if org_ids is not None:
            query += f"""
                WHERE OrganisationID IN ({', '.join(['%s'] * len(org_ids))})
            """

            self.cursor.execute(query, org_ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_concepts(self, concept_ids=None):

        self.connect()

        query = f"""
            SELECT PageID,
                   PageTitle
            FROM graph.Nodes_N_Concept
        """

        if concept_ids is not None:
            query += f"""
                WHERE PageID IN ({', '.join(['%s'] * len(concept_ids))})
            """

            self.cursor.execute(query, concept_ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    def get_crunchbase_concept_ids(self):

        self.connect()

        query = f"""
            SELECT DISTINCT PageID
            FROM graph.Edges_N_Organisation_N_Concept
        """

        self.cursor.execute(query)

        return [concept_id for concept_id, in self.cursor]







    # To be checked if deprecated
    def get_startups(self, ids=None, limit=None):

        self.connect()

        query = f"""
            SELECT EPFLStartupID,
                   StartupName,
                   LegalEntity,
                   FoundingYear,
                   ExitYear,
                   Status,
                   Industry
            FROM graph.Nodes_N_EPFLStartup
        """

        if ids is not None:
            query += f"""
                WHERE EPFLStartupID IN ({', '.join(['%s'] * len(ids))})
            """

        if limit is not None:
            query += f"""
                LIMIT {limit}
            """

        if ids is not None:
            self.cursor.execute(query, ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)

    # To be checked if deprecated
    def get_concepts_old(self, ids=None, limit=None):

        self.connect()

        query = f"""
            SELECT PageID,
                   PageTitle
            FROM graph.Nodes_N_Concept
        """

        if ids is not None:
            query += f"""
                WHERE PageID IN ({', '.join(['%s'] * len(ids))})
            """

        if limit is not None:
            query += f"""
                LIMIT {limit}
            """

        if ids is not None:
            self.cursor.execute(query, ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)








    # Deprecated, to be removed
    def get_funding_round_investors(self, fr_ids):

        self.connect()

        query = f"""
            SELECT FundingRoundID, OrganisationID, "organisation" as Type
            FROM graph.Edges_N_Organisation_N_FundingRound
            WHERE Action = "Invested in"
            AND FundingRoundID IN ({', '.join(['%s'] * len(fr_ids))})
        """

        self.cursor.execute(query, fr_ids)

        organisation_investors = list(self.cursor)

        query = f"""
            SELECT FundingRoundID, PersonID, "person" as Type
            FROM graph.Edges_N_Person_N_FundingRound
            WHERE Action = "Invested in"
            AND FundingRoundID IN ({', '.join(['%s'] * len(fr_ids))})
        """

        self.cursor.execute(query, fr_ids)

        person_investors = list(self.cursor)

        return organisation_investors + person_investors

    # Deprecated, to be removed
    def get_people(self, people_ids=None):

        self.connect()

        query = f"""
            SELECT PersonID,
                   FullName
            FROM graph.Nodes_N_Person
        """

        if people_ids is not None:
            query += f"""
                WHERE PersonID IN ({', '.join(['%s'] * len(people_ids))})
            """

        if people_ids is not None:
            self.cursor.execute(query, people_ids)
        else:
            self.cursor.execute(query)

        return list(self.cursor)
