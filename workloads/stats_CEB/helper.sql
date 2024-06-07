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