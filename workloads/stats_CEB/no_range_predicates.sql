SELECT COUNT(*) FROM  users as u, posts as p           WHERE  u.Id = p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, votes as v           WHERE  u.Id = v.UserId;
SELECT COUNT(*) FROM  users as u, badges as b          WHERE  u.Id = b.UserId;
SELECT COUNT(*) FROM  users as u, comments as c        WHERE  u.Id = c.UserId;
SELECT COUNT(*) FROM  users as u, postHistory as ph    WHERE  u.Id = ph.UserId; 
SELECT COUNT(*) FROM  posts as p, votes as v           WHERE  p.OwnerUserId = v.UserId; 
SELECT COUNT(*) FROM  posts as p, badges as b          WHERE  p.OwnerUserId = b.UserId; 
SELECT COUNT(*) FROM  posts as p, comments as c        WHERE  p.OwnerUserId = c.UserId;
SELECT COUNT(*) FROM  posts as p, postHistory as ph    WHERE  p.OwnerUserId = ph.UserId; 
SELECT COUNT(*) FROM  votes as v, badges as b          WHERE  v.UserId = b.UserId; 
SELECT COUNT(*) FROM  votes as v, comments as c        WHERE  v.UserId = c.UserId;
SELECT COUNT(*) FROM  votes as v, postHistory as ph    WHERE  v.UserId = ph.UserId;
SELECT COUNT(*) FROM  badges as b, comments as c       WHERE  b.UserId = c.UserId;
SELECT COUNT(*) FROM  badges as b, postHistory as ph   WHERE  b.UserId = ph.UserId;
SELECT COUNT(*) FROM  comments as c, postHistory as ph WHERE  c.UserId = ph.UserId;

SELECT COUNT(*) FROM  users as u, badges as b, comments as c         WHERE  u.Id = b.UserId AND u.Id=c.UserId;
SELECT COUNT(*) FROM  users as u, badges as b, votes as v            WHERE  u.Id = b.UserId AND u.Id=v.UserId;
SELECT COUNT(*) FROM  users as u, badges as b, posts as p            WHERE  u.Id = b.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, badges as b, postHistory as ph     WHERE  u.Id = b.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, comments as c, votes as v          WHERE  u.Id = c.UserId AND u.Id=v.UserId;
SELECT COUNT(*) FROM  users as u, comments as c, posts as p          WHERE  u.Id = c.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, comments as c, postHistory as ph   WHERE  u.Id = c.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, votes as v, posts as p             WHERE  u.Id = v.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, votes as v, postHistory as ph      WHERE  u.Id = v.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, posts as p, postHistory as ph      WHERE  u.Id = p.OwnerUserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  badges as b, comments as c, votes as v         WHERE  b.UserId=c.UserId AND b.UserId=v.UserId;
SELECT COUNT(*) FROM  badges as b, comments as c, posts as p         WHERE  b.UserId=c.UserId AND b.UserId=p.OwnerUserId;
SELECT COUNT(*) FROM  badges as b, comments as c, postHistory as ph  WHERE  b.UserId=c.UserId AND b.UserId=ph.UserId;
SELECT COUNT(*) FROM  badges as b, votes as v, posts as p            WHERE  b.UserId=v.UserId AND b.UserId=p.OwnerUserId;
SELECT COUNT(*) FROM  badges as b, votes as v, postHistory as ph     WHERE  b.UserId=v.UserId AND b.UserId=ph.UserId;
SELECT COUNT(*) FROM  badges as b, posts as p, postHistory as ph     WHERE  b.UserId=p.OwnerUserId AND b.UserId=ph.UserId;
SELECT COUNT(*) FROM  comments as c, votes as v, posts as p          WHERE  c.UserId=v.UserId AND c.UserId=p.OwnerUserId;
SELECT COUNT(*) FROM  comments as c, votes as v, postHistory as ph   WHERE  c.UserId=v.UserId AND c.UserId=ph.UserId;
SELECT COUNT(*) FROM  comments as c, posts as p, postHistory as ph   WHERE  c.UserId=p.OwnerUserId AND c.UserId=ph.UserId;
SELECT COUNT(*) FROM  votes as v, posts as p, postHistory as ph      WHERE  v.UserId=p.OwnerUserId AND v.UserId=ph.UserId;


SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v         WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId;
SELECT COUNT(*) FROM  users as u, badges as b, comments as c, posts as p         WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, badges as b, comments as c, postHistory as ph  WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, comments as c, votes as v,  posts as p         WHERE  u.Id = v.UserId AND u.Id=c.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, comments as c, votes as v,  postHistory as ph  WHERE  u.Id = v.UserId AND u.Id=c.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, votes as v,  posts as p, postHistory as ph     WHERE u.Id = v.UserId AND u.Id=p.OwnerUserId AND u.Id=ph.UserId;


SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v, posts as p           WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId AND u.Id=p.OwnerUserId;
SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v, postHistory as ph    WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, badges as b, votes as v, posts as p , postHistory as ph      WHERE  u.Id = b.UserId AND u.Id=p.OwnerUserId AND u.Id=v.UserId AND u.Id=ph.UserId;
SELECT COUNT(*) FROM  users as u, comments as c, votes as v, posts as p , postHistory as ph    WHERE  u.Id = c.UserId AND u.Id=p.OwnerUserId AND u.Id=v.UserId AND u.Id=ph.UserId;


SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v, posts as p, postHistory as ph    WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId AND u.Id=p.OwnerUserId  AND u.Id=ph.UserId;






