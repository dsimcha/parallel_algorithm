import std.typetuple, std.parallelism, std.range, std.functional,
    std.algorithm, std.stdio, std.array, std.traits, std.conv,
    core.stdc.string;

version(unittest) {
    import std.random, std.typecons, std.math;
}

private template finiteRandom(R) {
    enum bool finiteRandom = isRandomAccessRange!R && std.range.hasLength!R;
}

// Tracks whether the last merge was from -> buf or buf -> from.  This
// avoids needing to copy from buf to from after every iteration.
private enum MergedTo {
    from,
    buf
}

/**
Sort a range using a parallel merge sort algorithm, falling back to
$(D baseAlgorithm) for small subranges.  Usage is similar to
$(XREF algorithm, sort).

Params:

pred = The predicate to sort on.

baseAlgorithm = The algorithm to fall back to for small subranges.
$(D parallelSort) is a stable sort iff $(D baseAlgorithm) is a stable sort.

range = The range to be sorted.

minParallelSort = The smallest subrange to sort in parallel. Small values
will expose more parallelism, but also incur more overhead.

minParallelMerge = The smallest subrange to merge in parallel.  Since
merging is a cheaper operation than sorting, this should be somewhat larger
than $(D minParallelSort).

pool = The $(XREF parallelism, TaskPool) to use.  If null, the global
default task pool returned by $(XREF parallelism, taskPool) will be used.
*/
SortedRange!(R, pred)
parallelSort(alias pred = "a < b", alias baseAlgorithm = std.algorithm.sort, R)(
    R range,
    size_t minParallelSort = 1024,
    size_t minParallelMerge = 4096,
    TaskPool pool = null
) if(finiteRandom!R && hasAssignableElements!R) {
    // TODO:  Use C heap or TempAlloc or something.
    auto buf = new ElementType!(R)[range.length];

    if(pool is null) pool = std.parallelism.taskPool;

    immutable mergedTo = parallelSortImpl!(pred, baseAlgorithm, R, typeof(buf))
        (range, buf, minParallelSort, minParallelMerge, pool
    );

    if(mergedTo == MergedTo.buf) {
        copy(buf, range);
    }

    return SortedRange!(R, pred)(range);
}

unittest {
    // This algorithm is kind of complicated with all the tricks to prevent
    // excess copying and stuff.  Use monte carlo unit testing.

    auto gen = Random(314159265);  // Make tests deterministic but pseudo-random.
    foreach(i; 0..100) {
        auto nums = new uint[uniform(10, 20, gen)];
        foreach(ref num; nums) {
            num = uniform(0, 1_000_000, gen);
        }

        auto duped = nums.dup;
        parallelSort!"a > b"(duped, 4, 8);
        sort!"a > b"(nums);
        assert(duped == nums);
    }

    // Test sort stability.
    auto arr = new Tuple!(int, int)[32_768];
    foreach(ref elem; arr) {
        elem[0] = uniform(0, 10, gen);
        elem[1] = uniform(0, 10, gen);
    }

    static void stableSort(alias pred, R)(R range) {
        // Quick and dirty insertion sort, for testing only.
        alias binaryFun!pred comp;

        foreach(i; 1..range.length) {
            for(size_t j = i; j > 0; j--) {
                if(comp(range[j], range[j - 1])) {
                    swap(range[j], range[j - 1]);
                } else {
                    break;
                }
            }
        }
    }

    parallelSort!("a[1] < b[1]", stableSort)(arr);
    assert(isSorted!"a[1] < b[1]"(arr));
    parallelSort!("a[0] < b[0]", stableSort)(arr);
    assert(isSorted!"a[0] < b[0]"(arr));

    foreach(i; 0..arr.length - 1) {
        if(arr[i][0] == arr[i + 1][0]) {
            assert(arr[i][1] <= arr[i + 1][1]);
        }
    }
}

MergedTo parallelSortImpl(alias pred, alias baseAlgorithm, R1, R2)(
    R1 range,
    R2 buf,
    size_t minParallelSort,
    size_t minParallelMerge,
    TaskPool pool
) {
    assert(pool);

    if(range.length < minParallelSort) {
        baseAlgorithm!pred(range);
        return MergedTo.from;
    }

    immutable len = range.length;
    auto left = range[0..len / 2];
    auto right = range[len / 2..len];
    auto bufLeft = buf[0..len / 2];
    auto bufRight = buf[len / 2..len];

    auto ltask = scopedTask!(parallelSortImpl!(pred, baseAlgorithm, R1, R2))(
        left, bufLeft, minParallelSort, minParallelMerge, pool
    );
    pool.put(ltask);

    immutable rloc = parallelSortImpl!(pred, baseAlgorithm, R1, R2)(
        right, bufRight, minParallelSort, minParallelMerge, pool
    );

    auto lloc = ltask.yieldForce();

    if(lloc == MergedTo.from && rloc == MergedTo.buf) {
        copy(left, bufLeft);
        lloc = MergedTo.buf;
    } else if(lloc == MergedTo.buf && rloc == MergedTo.from) {
        copy(right, bufRight);
    }

    if(lloc == MergedTo.from) {
        parallelMerge!(pred, R1, R1, R2)(left, right, buf, minParallelMerge);
        return MergedTo.buf;
    } else {
        parallelMerge!(pred, R2, R2, R1)(bufLeft, bufRight, range, minParallelMerge);
        return MergedTo.from;
    }
}

/**
Merge ranges $(D from1) and $(D from2), which are assumed sorted according
to $(D pred), into $(D buf) using a parallel divide-and-conquer algorithm.

Params:

from1 = The first of the two sorted ranges to be merged.  This must be a
random access range with length.

from2 = The second of the two sorted ranges to be merged.  This must also
be a random access range with length and must have an identical element type to
$(D from1).

buf = The buffer to merge into.  This must be a random access range with
length equal to $(D from1.length + from2.length) and must have assignable
elements.

minParallel = The minimum merge size to parallelize.  Smaller values
create more parallel work units resulting in greater scalability but
increased overhead.

pool = The $(XREF parallelism, TaskPool) to use.  If null, the global
default task pool returned by $(XREF parallelism, taskPool) will be used.
*/
void parallelMerge(alias pred = "a < b", R1, R2, R3)(
    R1 from1,
    R2 from2,
    R3 buf,
    size_t minParallel = 4096,
    TaskPool pool = null
) if(allSatisfy!(finiteRandom, TypeTuple!(R1, R2, R3)) &&
   is(ElementType!R1 == ElementType!R2) &&
   is(ElementType!R2 == ElementType!R3) &&
   hasAssignableElements!R3
)
in {
    assert(from1.length + from2.length == buf.length);
} body {
    if(buf.length < minParallel) {
        return merge!(pred, R1, R2, R3)(from1, from2, buf);
    }

    immutable len1 = from1.length;
    immutable len2 = from2.length;

    if(len1 == 0 && len2 == 0) {
        return;
    }

    typeof(from1) left1, right1;
    typeof(from2) left2, right2;
    alias binaryFun!pred comp;

    if(len1 > len2) {
        auto mid1Index = len1 / 2;

        // This is necessary to make the sort stable:
        while(mid1Index > 0 && !comp(from1[mid1Index - 1], from1[mid1Index])) {
            mid1Index--;
        }

        auto mid1 = from1[mid1Index];
        left1 = from1[0..mid1Index];
        right1 = from1[mid1Index..len1];
        left2 = assumeSorted!pred(from2).lowerBound(mid1).release;
        right2 = from2[left2.length..len2];
    } else {
        auto mid2Index = len2 / 2;

        // This is necessary to make the sort stable:
        while(mid2Index > 0 && !comp(from2[mid2Index - 1], from2[mid2Index])) {
            mid2Index--;
        }

        auto mid2 = from2[mid2Index];
        left2 = from2[0..mid2Index];
        right2 = from2[mid2Index..len2];
        left1 = assumeSorted!pred(from1).lowerBound(mid2).release;
        right1 = from1[left1.length..len1];
    }

    auto leftBuf = buf[0..left1.length + left2.length];
    auto rightBuf = buf[leftBuf.length..buf.length];

    if(leftBuf.length == 0 || rightBuf.length == 0) {
        // Then recursing further would lead to infinite recursion.
        return merge!(pred, R1, R2, R3)(from1, from2, buf);
    }

    if(pool is null) pool = std.parallelism.taskPool;

    auto rightTask = scopedTask!(parallelMerge!(pred, R1, R2, R3))(
        right1, right2, rightBuf, minParallel, pool
    );

    pool.put(rightTask);
    parallelMerge!(pred, R1, R2, R3)(left1, left2, leftBuf, minParallel, pool);
    rightTask.yieldForce();
}

unittest {
    auto from1 = [1, 2, 4, 8, 16, 32];
    auto from2 = [2, 4, 6, 8, 10, 12];
    auto buf = new int[from1.length + from2.length];
    parallelMerge(from1, from2, buf, 2);
    assert(buf == [1, 2, 2, 4, 4, 6, 8, 8, 10, 12, 16, 32]);
}

/**
Merge ranges $(D from1) and $(D from2), which are assumed sorted according
to $(D pred), into $(D buf) using a sequential algorithm.

Params:

from1 = The first of the two sorted ranges to be merged.

from2 = The second of the two sorted ranges to be merged.  This must also
be an input range and must have an identical element type to
$(D from1).

buf = The buffer to merge into.  This must be an output range with
capacity at least equal to $(D walkLength(from1) + walkLength(from2)).

Example:
---
auto from1 = [1, 2, 4, 8, 16, 32];
auto from2 = [2, 4, 6, 8, 10, 12];
auto buf = new int[from1.length + from2.length];
merge(from1, from2, buf);
assert(buf == [1, 2, 2, 4, 4, 6, 8, 8, 10, 12, 16, 32]);
---
*/
void merge(alias pred = "a < b", R1, R2, R3)(
    R1 from1,
    R2 from2,
    R3 buf
) if(allSatisfy!(isInputRange, TypeTuple!(R1, R2)) &&
     is(ElementType!R1 == ElementType!R2) &&
     is(ElementType!R2 == ElementType!R3) &&
     isOutputRange!(R3, ElementType!R1)
) {
    alias binaryFun!(pred) comp;

    static if(allSatisfy!(isRandomAccessRange, TypeTuple!(R1, R2, R3))) {
        // This code is empirically slightly more efficient in the case of
        // arrays.
        size_t index1 = 0, index2 = 0, bufIndex = 0;
        immutable len1 = from1.length;
        immutable len2 = from2.length;

        while(index1 < len1 && index2 < len2) {
            if(comp(from2[index2], from1[index1])) {
                buf[bufIndex] = from2[index2];
                index2++;
            } else {
                buf[bufIndex] = from1[index1];
                index1++;
            }

            bufIndex++;
        }

        if(index1 < len1) {
            assert(index2 == len2);
            copy(from1[index1..len1], buf[bufIndex..len1 + len2]);
        } else if(index2 < len2) {
            assert(index1 == len1);
            copy(from2[index2..len2], buf[bufIndex..len1 + len2]);
        }
    } else {
        // Fall back to the obvious generic impl.
        while(!from1.empty && !from2.empty) {
            if(comp(from2.front, from1.front)) {
                buf.put(from2.front);
                from2.popFront();
            } else {
                buf.put(from1.front);
                from1.popFront();
            }
        }

        if(!from1.empty) {
            assert(from2.empty);
            copy(from1, buf);
        } else if(!from2.empty) {
            assert(from1.empty);
            copy(from2, buf);
        }
    }
}

unittest {
    auto from1 = [1, 2, 4, 8, 16, 32];
    auto from2 = [2, 4, 6, 8, 10, 12];
    auto buf = new int[from1.length + from2.length];
    merge(from1, from2, buf);
    assert(buf == [1, 2, 2, 4, 4, 6, 8, 8, 10, 12, 16, 32]);
}

CommonType!(ElementType!(Range1),ElementType!(Range2))
parallelDotProduct(Range1, Range2)(
    Range1 a,
    Range2 b,
    TaskPool pool = null,
    size_t workUnitSize = size_t.max
) if(isFloatingPoint!(ElementType!Range1) && isFloatingPoint!(ElementType!Range2)
   && isRandomAccessRange!Range1 && isRandomAccessRange!Range2
   && hasLength!Range1 && hasLength!Range2
) in {
    assert(a.length == b.length);
} body {

    if(pool is null) pool = taskPool;
    if(workUnitSize == size_t.max) {
        workUnitSize = pool.defaultWorkUnitSize(a.length);
    }

    alias typeof(return) F;

    // A TaskPool.reduce/std.algorithm.map approach seems like the obvious one
    // here, but std.numeric.dotProduct is so well optimized that I can't beat
    // it unless I use it under the hood.  Create a range of dot products of
    // slices, and let TaskPool.reduce handle the minutiae of sending each
    // element to the task pool.  Eventually this slice-reduce paradigm
    // needs to be extracted into reusable code.
    static struct SliceDotProd {
        Range1 a;
        Range2 b;
        size_t workUnitSize;

        // These primitives are only needed to make this range
        // pass the isRandomAccess test, but are never actually used.  Just make
        // them stubs.
        F front() @property { assert(0); }
        void popFront() { assert(0); }
        F back() @property { assert(0); }
        void popBack() @property { assert(0); }
        typeof(this) opSlice(size_t foo, size_t bar) { assert(0); }
        bool empty() @property { assert(0); }

        typeof(this) save() { return typeof(this)(a, b, workUnitSize); }

        size_t length() @property {
            assert(a.length == b.length);
            return (a.length / workUnitSize) + (a.length % workUnitSize > 0);
        }

        F opIndex(size_t index) {
            immutable start = workUnitSize * index;
            immutable stop = min(a.length, workUnitSize * (index + 1));
            assert(start <= stop, text(start, ' ', stop));
            return std.numeric.dotProduct(a[start..stop], b[start..stop]);
        }
    }

    return taskPool.reduce!"a + b"(cast(F) 0, SliceDotProd(a, b, workUnitSize), 1);
}

unittest {
    auto a = new double[10_000];
    auto b = new double[10_000];

    foreach(i, ref elem; a) {
        elem = i;
        b[i] = i;
    }

    assert(approxEqual(std.numeric.dotProduct(a, b), parallelDotProduct(a, b, taskPool)));
}


//////////////////////////////////////////////////////////////////////////////
// Benchmarks
//////////////////////////////////////////////////////////////////////////////
import std.random, std.datetime, std.exception, std.numeric;

void mergeBenchmark() {
    enum N = 8192;
    enum nIter = 1000;
    auto a = new float[N];
    auto b = new float[N];
    auto buf = new float[a.length + b.length];

    foreach(ref elem; chain(a, b)) elem = uniform(0f, 1f);
    sort(a);
    sort(b);

    auto sw = StopWatch(AutoStart.yes);
    foreach(i; 0..nIter) merge(a, b, buf);
    writeln("Serial Merge:  ", sw.peek.msecs);
    assert(equal(buf, sort(a ~ b)));

    sw.reset();
    foreach(i; 0..nIter) parallelMerge(a, b, buf, 2048);
    writeln("Parallel Merge:  ", sw.peek.msecs);
    assert(equal(buf, sort(a ~ b)));
}

void sortBenchmark() {
    enum N = 32768;
    enum nIter = 100;
    auto a = new ushort[N];
    foreach(ref elem; a) elem = uniform(cast(ushort) 0, ushort.max);

    auto sw = StopWatch(AutoStart.yes);
    sort(a);
    writeln("Serial Sort:    ", sw.peek.usecs);
    enforce(isSorted(a));

    randomShuffle(a);
    sw.reset();
    parallelSort!("a < b", sort)(a, 4096, 4096);
    writeln("Parallel Sort:  ", sw.peek.usecs);
    enforce(isSorted(a));
}

void dotProdBenchmark() {
    enum n = 100_000;
    enum nIter = 100;
    auto a = new float[n];
    auto b = new float[n];

    foreach(ref num; chain(a, b)) {
        num = uniform(0.0, 1.0);
    }

    auto sw = StopWatch(AutoStart.yes);
    foreach(i; 0..nIter) std.numeric.dotProduct(a, b);
    writeln("Serial dot product:    ", sw.peek.msecs);

    sw.reset();
    foreach(i; 0..nIter) parallelDotProduct(a, b, taskPool);
    writeln("Parallel dot product:  ", sw.peek.msecs);
}


void main() {
    mergeBenchmark();
    sortBenchmark();
    dotProdBenchmark();
}
