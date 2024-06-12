SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v, posts as p           WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v, postHistory as ph    WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, badges as b, votes as v, posts as p , postHistory as ph      WHERE  u.Id = b.UserId AND u.Id=p.OwnerUserId AND u.Id=v.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, comments as c, votes as v, posts as p , postHistory as ph    WHERE  u.Id = c.UserId AND u.Id=p.OwnerUserId AND u.Id=v.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  posts as p, postLinks as pl, comments as c, postHistory as ph, votes as v      WHERE  p.Id = pl.PostId AND p.Id = c.PostId AND p.Id = ph.PostId AND p.Id = v.PostId;
SELECT COUNT(*) FROM  posts as p, postLinks as pl, comments as c, postHistory as ph, votes as v      WHERE  p.Id = pl.RelatedPostId AND p.Id = c.PostId AND p.Id = ph.PostId AND p.Id = v.PostId;