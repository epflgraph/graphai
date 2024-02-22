CREATE SCHEMA IF NOT EXISTS `auth_graphai` ;

CREATE TABLE IF NOT EXISTS `auth_graphai`.`Users` (
  `username` varchar(255) NOT NULL,
  `full_name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `hashed_password` varchar(255) NOT NULL,
  `disabled` tinyint(1) NOT NULL DEFAULT 0,
  `scopes` varchar(255) NULL DEFAULT 'user',
  PRIMARY KEY (`username`),
  KEY `full_name` (`full_name`),
  KEY `email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- You can hash your chosen password using graphai.core.common.auth_utils.get_password_hash before inserting it below
-- The scopes column should contain a comma-separated string of the scopes that the user is allowed to access.
-- We recommend using the full list included below for the admin user, and giving more restricted access to other users.

INSERT INTO `auth_graphai`.`Users`
(`username`,
`full_name`,
`email`,
`hashed_password`,
`disabled`,
`scopes`)
VALUES
('admin',
'GraphAI Admin',
'your.email@provider.com',
'yourbcrypthashedpassword',
0,
'user,voice,video,translation,text,scraping,ontology,image,completion');
