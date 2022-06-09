select *
from PortofolioProject..ResultCrawl --where tweet is not null and polarity is not null

-- delete empy rows
delete from PortofolioProject..ResultCrawl where tweet IS NULL and polarity is null

-- remove @ character
select
SUBSTRING(tweet, 3, LEN(tweet)) as NewTweet
from PortofolioProject..ResultCrawl

SELECT REPLACE(REPLACE(tweet, '@', ''), CHAR(13), '')
from PortofolioProject..ResultCrawl

update PortofolioProject..ResultCrawl
set tweet = SUBSTRING(tweet, 3, LEN(tweet))

update PortofolioProject..ResultCrawl
set tweet = REPLACE(REPLACE(tweet, '@', ''), CHAR(13), '')
