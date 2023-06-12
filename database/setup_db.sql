--- Create extensions
CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;
COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;
COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';

--- Create st extend function
-- References:
-- http://blog.cleverelephant.ca/2015/02/breaking-linestring-into-segments.html
-- https://gis.stackexchange.com/a/104451/44921
-- https://gis.stackexchange.com/a/16701/44921
CREATE FUNCTION public.st_extend(geom public.geometry, head_rate double precision, head_constant double precision, tail_rate double precision, tail_constant double precision) RETURNS public.geometry
    LANGUAGE sql
    AS $_$
WITH segment_parts AS (
SELECT
(pt).path[1]-1 as segment_num
,
CASE
WHEN
  (nth_value((pt).path, 2) OVER ()) = (pt).path
AND
  (last_value((pt).path) OVER ()) = (pt).path
THEN
  3
WHEN
  (nth_value((pt).path, 2) OVER ()) = (pt).path
THEN
  1
WHEN
  (last_value((pt).path) OVER ()) = (pt).path
THEN
  2
ELSE
  0
END AS segment_flag
,
(pt).geom AS a
,
lag((pt).geom, 1, NULL) OVER () AS b
FROM ST_DumpPoints($1) pt
)
,
extended_segment_parts
AS
(
SELECT
  *
  ,
  ST_Azimuth(a,b) AS az1
  ,
  ST_Azimuth(b,a) AS az2
  ,
  ST_Distance(a,b) AS len
FROM
segment_parts
where b IS NOT NULL
)
,
expanded_segment_parts
AS
(
SELECT
  segment_num
  ,
  CASE
  WHEN
    bool(segment_flag & 2)
  THEN
    ST_Translate(b, sin(az2) * (len*tail_rate+tail_constant), cos(az2) * (len*tail_rate+tail_constant))
  ELSE
    a
  END
  AS a
  ,
  CASE
  WHEN
    bool(segment_flag & 1)
  THEN
    ST_Translate(a, sin(az1) * (len*head_rate+head_constant), cos(az1) * (len*head_rate+head_constant))
  ELSE
    b
  END
  AS b
FROM extended_segment_parts
)
,
expanded_segment_lines
AS
(
SELECT
  segment_num
  ,
  ST_MakeLine(a, b) as geom
FROM
expanded_segment_parts
)
SELECT
  ST_LineMerge(ST_Collect(geom ORDER BY segment_num)) AS geom
FROM expanded_segment_lines
;
$_$;


ALTER FUNCTION public.st_extend(geom public.geometry, head_rate double precision, head_constant double precision, tail_rate double precision, tail_constant double precision) OWNER TO postgres;


---  Create sequences
CREATE SEQUENCE public.accepted_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

SET default_tablespace = '';
SET default_table_access_method = heap;

--- Create accepted towers table for OSM

CREATE TABLE public.accepted_towers (
    id integer DEFAULT nextval('public.accepted_id_seq'::regclass) NOT NULL,
    tower_uuid uuid,
    run_id integer,
    tmp_line_id uuid
);


--- Create main powertowers table

CREATE TABLE public.powertower (
    id integer NOT NULL,
    tile_id character varying,
    score double precision NOT NULL,
    bought_tiles integer,
    spent_budget double precision,
    object_type character varying,
    run_id integer,
    tmp_line_id uuid,
    geom public.geometry(Point,4326),
    tower_uuid uuid DEFAULT public.uuid_generate_v4(),
    cost double precision
);

CREATE SEQUENCE public.powertower_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.powertower_id_seq OWNED BY public.powertower.id;


CREATE TABLE public.predicted (
    tile_id character varying NOT NULL,
    run_id integer NOT NULL,
    id integer NOT NULL
);


CREATE SEQUENCE public.predicted_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.predicted_id_seq OWNED BY public.predicted.id;


CREATE TABLE public.run (
    id integer NOT NULL,
    description character varying
);


CREATE SEQUENCE public.run_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.run_id_seq OWNED BY public.run.id;


CREATE TABLE public.startingpoints (
    id integer NOT NULL,
    id_column character varying,
    object_id character varying,
    file_path character varying,
    set_name character varying,
    geom public.geometry(Point,4326)
);


CREATE SEQUENCE public.startingpoints_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.startingpoints_id_seq OWNED BY public.startingpoints.id;

CREATE TABLE public.sublines (
    id uuid NOT NULL,
    parent_line uuid,
    split_from uuid,
    starting_tower uuid,
    startingpoint integer,
    run_id integer,
    mean_score double precision,
    num_towers integer,
    master_line uuid,
    geom public.geometry(LineString,4326)
);


CREATE TABLE public.subsublines (
    id uuid NOT NULL,
    parent_subline uuid,
    starting_tower uuid,
    run_id integer,
    mean_score double precision,
    num_towers integer,
    master_line uuid,
    geometry public.geometry(LineString,4326),
    batch_id character varying,
    mean_cost double precision,
    mean_direction double precision
);


CREATE SEQUENCE public.subsublines_towers_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    MAXVALUE 3323212313
    CACHE 1;


CREATE TABLE public.subsublines_towers (
    id integer DEFAULT nextval('public.subsublines_towers_id_seq'::regclass) NOT NULL,
    tower_uuid uuid,
    subsubline uuid,
    batch_id character varying,
    accepted boolean
);

CREATE TABLE public.tile (
    id character varying NOT NULL,
    x integer,
    y integer,
    file_path character varying,
    geom public.geometry(Polygon,4326)
);


ALTER TABLE ONLY public.powertower ALTER COLUMN id SET DEFAULT nextval('public.powertower_id_seq'::regclass);

ALTER TABLE ONLY public.predicted ALTER COLUMN id SET DEFAULT nextval('public.predicted_id_seq'::regclass);

ALTER TABLE ONLY public.run ALTER COLUMN id SET DEFAULT nextval('public.run_id_seq'::regclass);

ALTER TABLE ONLY public.startingpoints ALTER COLUMN id SET DEFAULT nextval('public.startingpoints_id_seq'::regclass);

ALTER TABLE ONLY public.accepted_towers
    ADD CONSTRAINT accepted_run_id_tow_key UNIQUE (run_id, tower_uuid);

ALTER TABLE ONLY public.accepted_towers
    ADD CONSTRAINT accepted_towers_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.subsublines
    ADD CONSTRAINT id_unique UNIQUE (id);

ALTER TABLE ONLY public.powertower
    ADD CONSTRAINT powertower_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.powertower
    ADD CONSTRAINT powertower_run_id_geom_key UNIQUE (run_id, geom);

ALTER TABLE ONLY public.powertower
    ADD CONSTRAINT powertower_tower_uuid_key UNIQUE (tower_uuid);

ALTER TABLE ONLY public.predicted
    ADD CONSTRAINT predicted_pkey PRIMARY KEY (tile_id, run_id);

ALTER TABLE ONLY public.run
    ADD CONSTRAINT run_pkey PRIMARY KEY (id);


ALTER TABLE ONLY public.startingpoints
    ADD CONSTRAINT startingpoints_id_column_object_id_file_path_set_name_key UNIQUE (id_column, object_id, file_path, set_name);


ALTER TABLE ONLY public.startingpoints
    ADD CONSTRAINT startingpoints_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.sublines
    ADD CONSTRAINT sublines_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.subsublines
    ADD CONSTRAINT subsublines_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.subsublines_towers
    ADD CONSTRAINT subsublines_towers_pk PRIMARY KEY (id);

ALTER TABLE ONLY public.tile
    ADD CONSTRAINT tile_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.subsublines_towers
    ADD CONSTRAINT uq_tower_batch UNIQUE (tower_uuid, batch_id);


CREATE INDEX powertower_geom_idx ON public.powertower USING gist (geom);


CREATE INDEX powertower_run_id_idx ON public.powertower USING btree (run_id);

CREATE INDEX powertower_tile_id_idx ON public.powertower USING btree (tile_id);

CREATE INDEX powertower_tmp_line_id_idx ON public.powertower USING btree (tmp_line_id);

CREATE INDEX startingpoints_geom_idx ON public.startingpoints USING gist (geom);

CREATE INDEX subsublines_geom_idx ON public.subsublines USING gist (geometry);

ALTER TABLE ONLY public.powertower
    ADD CONSTRAINT powertower_fk_tmp_line_id FOREIGN KEY (tmp_line_id) REFERENCES public.sublines(id) NOT VALID;

ALTER TABLE ONLY public.powertower
    ADD CONSTRAINT powertower_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.run(id);

ALTER TABLE ONLY public.powertower
    ADD CONSTRAINT powertower_tile_id_fkey FOREIGN KEY (tile_id) REFERENCES public.tile(id);


ALTER TABLE ONLY public.accepted_towers
    ADD CONSTRAINT powertower_tower_uuid_fkey FOREIGN KEY (tower_uuid) REFERENCES public.powertower(tower_uuid);


ALTER TABLE ONLY public.predicted
    ADD CONSTRAINT predicted_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.run(id);

ALTER TABLE ONLY public.predicted
    ADD CONSTRAINT predicted_tile_id_fkey FOREIGN KEY (tile_id) REFERENCES public.tile(id);

ALTER TABLE ONLY public.sublines
    ADD CONSTRAINT sublines_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.run(id);


ALTER TABLE ONLY public.sublines
    ADD CONSTRAINT sublines_startingpoint_fkey FOREIGN KEY (startingpoint) REFERENCES public.startingpoints(id);

ALTER TABLE ONLY public.subsublines
    ADD CONSTRAINT subsublines_parent_subline_fkey FOREIGN KEY (parent_subline) REFERENCES public.sublines(id);


ALTER TABLE ONLY public.subsublines
    ADD CONSTRAINT subsublines_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.run(id);

ALTER TABLE ONLY public.subsublines_towers
    ADD CONSTRAINT subsublines_towers_fk_tower_uuid FOREIGN KEY (tower_uuid) REFERENCES public.powertower(tower_uuid);