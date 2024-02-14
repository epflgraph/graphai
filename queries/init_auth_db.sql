CREATE SCHEMA IF NOT EXISTS `auth_graphai` ;

CREATE TABLE IF NOT EXISTS `auth_graphai`.`Users` (
  `username` varchar(255) NOT NULL,
  `full_name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `hashed_password` varchar(255) NOT NULL,
  `disabled` tinyint(1) NOT NULL DEFAULT 0,
  PRIMARY KEY (`username`),
  KEY `full_name` (`full_name`),
  KEY `email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `auth_graphai`.`Users`
(`username`,
`full_name`,
`email`,
`hashed_password`,
`disabled`)
VALUES
('admin',
'GraphAI Admin',
'your.email@provider.com',
'yourbcrypthashedpassword',
0);
