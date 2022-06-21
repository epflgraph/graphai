import mysql.connector
import configparser

from definitions import CONFIG_DIR

from concept_detection.test.types import WikifyResult
from concept_detection.text.html_cleaner import HTMLCleaner


class DB:
    def __init__(self):
        # Read db config from file and open connection
        db_config = configparser.ConfigParser()
        db_config.read(f'{CONFIG_DIR}/db.ini')
        self.cnx = mysql.connector.connect(host=db_config['DB'].get('host'), port=db_config['DB'].getint('port'), user=db_config['DB'].get('user'), password=db_config['DB'].get('password'))
        self.cursor = self.cnx.cursor()

        # Fetch initial data
        self.channel_anchor_page_ids = self.query_channel_anchor_page_ids()
        self.course_channel_ids = self.query_course_channel_ids()

    def query(self, query):
        self.cursor.execute(query)

        return list(self.cursor)

    def query_channel_anchor_page_ids(self):
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
        channel_id = slide_id.split('_')[0]
        return self.channel_anchor_page_ids[channel_id]

    def query_course_descriptions(self):
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
            c = HTMLCleaner()
            c.feed(course_description)
            course_descriptions[course_id] = c.get_data()

        return course_descriptions

    def query_wikified_course_descriptions(self):
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
        channel_ids = self.course_channel_ids.get(course_id, None)
        if channel_ids:
            return list({anchor_page_id for channel_id in channel_ids for anchor_page_id in self.channel_anchor_page_ids[channel_id]})
        else:
            return []

    def get_wikipage_ids(self):
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
        query = """
            SELECT titles.PageID, titles.PageTitle, contents.PageContent
            FROM piper_wikipedia.PageTitle_to_PageID_Mapping AS titles
            INNER JOIN piper_wikipedia.Page_Content AS contents
            ON titles.PageID = contents.PageID
        """

        if ids is not None:
            ids = [str(_id) for _id in ids]
            query += f"""
                WHERE titles.PageID IN ({','.join(ids)})
            """
        elif id_min_max is not None:
            query += f"""
                WHERE titles.PageID >= {id_min_max[0]}
                AND titles.PageID < {id_min_max[1]}
            """

        if limit is not None:
            query += f"""
                LIMIT {limit}
            """

        self.cursor.execute(query)

        return {
            _id: {
                'title': title,
                'content': content
            }
            for _id, title, content in self.cursor
        }

    def get_wikipage_categories(self, ids=None, id_min_max=None, limit=None):
        query = """            
            SELECT titles.PageID, GROUP_CONCAT(categories.CategoryTitle)
            FROM piper_wikipedia.PageTitle_to_PageID_Mapping AS titles
            INNER JOIN piper_wikipedia.PageID_to_CategoriesID_Mapping AS pages_categories
            ON titles.PageId = pages_categories.PageID
            INNER JOIN piper_wikipedia.Categories AS categories
            ON pages_categories.CategoryID = categories.CategoryID
        """

        if ids is not None:
            ids = [str(_id) for _id in ids]
            query += f"""
                WHERE pages_categories.PageID IN ({','.join(ids)})
            """
        elif id_min_max is not None:
            query += f"""
                WHERE pages_categories.PageID >= {id_min_max[0]}
                AND pages_categories.PageID < {id_min_max[1]}
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
