"""AI Hub 분할 zip(TS.z01 ... TS.z05 + TS.zip)을 합치지 않고 읽는다.

zipfile은 여러 디스크로 나뉜 ZIP64를 열지 못하고(spanning 미지원), 합치려면 588GB가 더 필요하다.
마지막 볼륨(.zip)의 중앙 디렉터리를 직접 읽어, 파일마다 (시작 볼륨, 볼륨 안 위치)를 전체 위치로 바꿔 읽는다.
    z = SplitZip('/.../TS.zip'); names = z.namelist(); b = z.read(names[0])
마지막 볼륨(.zip)을 아직 받지 못했으면 받은 .zNN 볼륨의 로컬 헤더를 차례로 훑어 끝까지 들어 있는 파일만 목록에 넣는다.
"""
import glob, os, re, struct, zlib


class SplitZip:
    def __init__(self, last):
        stem = last[:-4]
        vols = sorted(glob.glob(glob.escape(stem) + '.z[0-9][0-9]'), key=lambda p: int(re.search(r'\.z(\d+)$', p).group(1)))
        self.complete = os.path.exists(last)
        self.vols = vols + [last] if self.complete else vols
        self.sizes = [os.path.getsize(p) for p in self.vols]
        self.cum = [0]
        for s in self.sizes:
            self.cum.append(self.cum[-1] + s)
        self.fh = [None] * len(self.vols)
        self._read_cd() if self.complete else self._scan_local()
        self.close()

    def close(self):
        """열린 파일을 닫는다 (fork 전에 불러 두면 프로세스마다 따로 연다)."""
        for f in self.fh:
            if f is not None:
                f.close()
        self.fh = [None] * len(self.vols)

    def _scan_local(self):
        self.info = {}; pos = 4 if self._pread(0, 4) == b'PK\x07\x08' else 0; end = self.cum[-1]
        while pos + 30 <= end:
            h = self._pread(pos, 30)
            if h[:4] != b'PK\x03\x04':
                break
            _, _, flag, method, _, _, crc, csize, usize, nlen, elen = struct.unpack('<IHHHHHIIIHH', h)
            if pos + 30 + nlen + elen > end:
                break
            name = self._pread(pos + 30, nlen).decode('utf-8', 'replace'); ex = self._pread(pos + 30 + nlen, elen); i = 0
            while i + 4 <= len(ex):
                tag, sz = struct.unpack('<HH', ex[i:i + 4])
                if tag == 1:
                    vals = list(struct.unpack('<' + 'Q' * (sz // 8), ex[i + 4:i + 4 + sz // 8 * 8]))
                    if usize == 0xFFFFFFFF and vals: usize = vals.pop(0)
                    if csize == 0xFFFFFFFF and vals: csize = vals.pop(0)
                i += 4 + sz
            if flag & 8:
                raise ValueError('data descriptor: size unknown')
            if pos + 30 + nlen + elen + csize > end:
                break
            self.info[name] = (pos, method, csize, usize, crc)
            pos += 30 + nlen + elen + csize

    def _pread(self, pos, n):
        """전체 위치 pos에서 n바이트 (볼륨 경계를 넘을 수 있다)."""
        out = []
        while n > 0:
            v = max(i for i in range(len(self.vols)) if self.cum[i] <= pos)
            if self.fh[v] is None:
                self.fh[v] = open(self.vols[v], 'rb')
            f = self.fh[v]; f.seek(pos - self.cum[v])
            k = min(n, self.cum[v + 1] - pos)
            b = f.read(k)
            if len(b) != k:
                raise IOError(f'short read in {self.vols[v]}')
            out.append(b); pos += k; n -= k
        return b''.join(out)

    def _read_cd(self):
        last = len(self.vols) - 1
        tail_n = min(self.sizes[last], 1 << 20)
        tail = self._pread(self.cum[last + 1] - tail_n, tail_n)
        e = tail.rfind(b'PK\x05\x06')
        if e < 0:
            raise ValueError('no end of central directory')
        _, disk, cd_disk, _, n_tot, cd_size, cd_off, _ = struct.unpack('<IHHHHIIH', tail[e:e + 22])
        loc = tail.rfind(b'PK\x06\x07', 0, e)
        if loc >= 0:                                   # ZIP64
            _, z64_disk, z64_off, _ = struct.unpack('<IIQI', tail[loc:loc + 20])
            r = self._pread(self.cum[z64_disk] + z64_off, 56)
            if r[:4] != b'PK\x06\x06':
                raise ValueError('bad zip64 end record')
            _, _, _, _, disk, cd_disk, _, n_tot, cd_size, cd_off = struct.unpack('<IQHHIIQQQQ', r)
        cd = self._pread(self.cum[cd_disk] + cd_off, cd_size)
        self.info = {}; p = 0
        for _ in range(n_tot):
            if cd[p:p + 4] != b'PK\x01\x02':
                raise ValueError(f'bad central header at {p}')
            (method, crc, csize, usize, nlen, elen, clen, dstart, off) = (
                struct.unpack('<H', cd[p + 10:p + 12])[0], *struct.unpack('<III', cd[p + 16:p + 28]),
                *struct.unpack('<HHHH', cd[p + 28:p + 36]), struct.unpack('<I', cd[p + 42:p + 46])[0])
            name = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
            ex = cd[p + 46 + nlen:p + 46 + nlen + elen]; i = 0
            while i + 4 <= len(ex):
                tag, sz = struct.unpack('<HH', ex[i:i + 4]); body = ex[i + 4:i + 4 + sz]; j = 0
                if tag == 1:
                    if usize == 0xFFFFFFFF: usize = struct.unpack('<Q', body[j:j + 8])[0]; j += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack('<Q', body[j:j + 8])[0]; j += 8
                    if off == 0xFFFFFFFF: off = struct.unpack('<Q', body[j:j + 8])[0]; j += 8
                    if dstart == 0xFFFF: dstart = struct.unpack('<I', body[j:j + 4])[0]; j += 4
                i += 4 + sz
            self.info[name] = (self.cum[dstart] + off, method, csize, usize, crc)
            p += 46 + nlen + elen + clen

    def namelist(self):
        return list(self.info)

    def read(self, name):
        pos, method, csize, usize, crc = self.info[name]
        h = self._pread(pos, 30)
        if h[:4] != b'PK\x03\x04':
            raise ValueError(f'bad local header for {name}')
        nlen, elen = struct.unpack('<HH', h[26:30])
        b = self._pread(pos + 30 + nlen + elen, csize)
        if method == 8:
            b = zlib.decompress(b, -15)
        elif method != 0:
            raise ValueError(f'method {method}')
        if zlib.crc32(b) & 0xFFFFFFFF != crc:
            raise ValueError(f'CRC mismatch for {name}')
        return b
