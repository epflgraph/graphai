from graphai.core.common.caching import VideoDBCachingManager


def main():
    db_manager = VideoDBCachingManager()
    try:
        db_manager.add_columns(db_manager.cache_table, ["date_modified"], ["DATETIME"], ["NULL"])
        success = True
    except Exception:
        success = False
    if success:
        print('Successful, now moving date_added to date_modified')
        db_manager.db.execute_query("UPDATE `%s`.`%s` "
                                    "SET date_modified = date_added",
                                    values=(db_manager.schema, db_manager.cache_table))


if __name__ == '__main__':
    main()
