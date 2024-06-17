-- U.Id=b.UserId=c.UserId=ph.UserId=v.UserId=p.OwnerUserId
select min(id),max(id) from users;
select min(userid),max(userid) from badges;
select min(userid),max(userid) from comments;
select min(userid),max(userid) from postHistory;
select min(userid),max(userid) from votes;
select min(OwnerUserId),max(OwnerUserId) from posts;
-- final is [-1, 55747]


-- p.Id=pl.PostId=c.PostId=ph.PostId=v.PostId=t.ExcerptPostId
--     =pl.RelatedPostId
select min(id),max(id) from posts;
--  [1 , 115378]



select u.Id, count(*)  as cnt from users as u group by 1 order by 2 desc;

select u.Id, count(*)  as cnt from users as u, badges as b where u.id=b.userid group by 1 order by 2 desc;

select u.Id, count(*)  as cnt from users as u, badges as b, comments as c where u.id=b.userid and u.id=c.userid group by 1 order by 2 desc;

select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, postHistory as ph where u.id=b.userid and u.id=c.userid and u.id=ph.userid group by 1 order by 2 desc;

select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, postHistory as ph, votes as v where u.id=b.userid and u.id=c.userid and u.id=ph.userid and u.id=v.userid group by 1 order by 2 desc;



select sum(cnt) from 
(
select u.Id, count(*) as cnt from users as u group by 1 order by 2 DESC  LIMIT 1000
) t;
1
10
100
1000

40325

select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b where u.id=b.userid group by 1 order by 2 DESC LIMIT 1000
) t;
456
2316
7953
23463

79851

select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b, comments as c where u.id=b.userid and u.id=c.userid  group by 1 order by 2 DESC LIMIT 1000
) t;
6028320
12207751
14874968
15702975

15900001


select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, votes as v where u.id=b.userid and u.id=c.userid and u.id=v.userid  group by 1 order by 2 DESC LIMIT 1
) t;
select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, votes as v where u.id=b.userid and u.id=c.userid and u.id=v.userid  group by 1 order by 2 DESC LIMIT 10
) t;
select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, votes as v where u.id=b.userid and u.id=c.userid and u.id=v.userid  group by 1 order by 2 DESC LIMIT 100
) t;
select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, votes as v where u.id=b.userid and u.id=c.userid and u.id=v.userid  group by 1 order by 2 DESC LIMIT 1000
) t;
1427299116
2703632552
2800877453
2809920313

2810041173

select sum(cnt) from 
(
select u.Id, count(*)  as cnt from users as u, badges as b, comments as c, postHistory as ph, votes as v where u.id=b.userid and u.id=c.userid and u.id=ph.userid and u.id=v.userid group by 1 order by 2 DESC  LIMIT 1
) t;
